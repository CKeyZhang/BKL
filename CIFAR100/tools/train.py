import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from mdistiller.engine.temperature_scaling import ModelWithTemperature
from datetime import datetime

def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == 'MLKD':
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    if cfg.DISTILLER.TYPE == "AKD" and cfg.AKD.TEMPSCALE == True:
        m_model_teacher = net(num_classes=num_classes)
        m_model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
        # Choose base data loder
        data_loader = train_loader

        # Now we're going to wrap the model with a decorator that adds temperature scaling
        model = ModelWithTemperature(m_model_teacher)

        # Tune the model temperature, and save the results
        model.set_temperature(data_loader)
        cfg.defrost()
        cfg.AKD.SCALETEMP = model.temperature.item()
        cfg.freeze()

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logit_stand", action="store_true")
    parser.add_argument("--base_temp", type=float, default=2)
    parser.add_argument("--kd_weight", type=float, default=9)
    parser.add_argument("--temp", type=float, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--kl_type", type=float, default='bkl')
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    if args.logit_stand and cfg.DISTILLER.TYPE in ['KD','DKD','MLKD']:
        cfg.EXPERIMENT.LOGIT_STAND = True
        if cfg.DISTILLER.TYPE == 'KD':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
            cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
            cfg.DKD.T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'MLKD':
            cfg.MLKD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.MLKD.TEMPERATURE = args.base_temp

    cfg.KD.TEMPERATURE = args.temp
    cfg.KD.BETA = args.beta
    cfg.KD.ALPHA = args.alpha
    cfg.KD.WARMUP = args.warmup

    today = datetime.today()
    formatted_date = today.strftime("%m%d")
    cfg.LOG.PREFIX = os.path.join(cfg.LOG.PREFIX, formatted_date)

    cfg.freeze()
    main(cfg, args.resume, args.opts)
