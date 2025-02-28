import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

from ._base import Distiller


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

def rakd_loss(logits_student, logits_teacher, alpha, beta, temperature):
    log_pred_teacher = F.log_softmax(logits_teacher / temperature, dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    loss_fkd = F.kl_div(log_pred_student, pred_teacher, reduction="none")
    loss_rkd = F.kl_div(log_pred_teacher, pred_student, reduction="none")

    rho = ((pred_student ** 2) + (pred_teacher ** 2)) ** (0.5 * (1 - beta))

    epsilon = 1e-12
    loss_part1 = (
        (loss_fkd / (rho + epsilon)) ** 2
    ).sum(1)
    loss_part2 = (
        (loss_rkd / (rho + epsilon)) ** 2
    ).sum(1)

    loss_all = alpha * loss_part1 + (1 - alpha) * loss_part2

    loss_akd = (
        loss_all.sum(0)
        / logits_student.shape[0]
    )
    loss_akd *= temperature ** 2
    
    return loss_akd


class ReviewKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ReviewKD, self).__init__(student, teacher)
        self.shapes = cfg.REVIEWKD.SHAPES
        self.out_shapes = cfg.REVIEWKD.OUT_SHAPES
        in_channels = cfg.REVIEWKD.IN_CHANNELS
        out_channels = cfg.REVIEWKD.OUT_CHANNELS
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.kd_loss_weight = cfg.REVIEWKD.KD_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL
        self.temperature = 8.0
        self.beta = 0.2
        self.alpha = 0.5

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # 打印学生和教师模型的特征图形状
        print("Student Features (preact_feats):", [f.shape for f in features_student["preact_feats"]])
        print("Student Pooled Feature:", features_student["pooled_feat"].shape)
        
        print("Teacher Features (preact_feats):", [f.shape for f in features_teacher["preact_feats"]])
        print("Teacher Pooled Feature:", features_teacher["pooled_feat"].shape)

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_reviewkd = (
            self.reviewkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * hcl_loss(results, features_teacher)
        )
        loss_bkl = self.kd_loss_weight * rakd_loss(
                logits_student, logits_teacher, self.alpha, self.beta, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce + loss_bkl,
            "loss_kd": loss_reviewkd,
        }
        return logits_student, losses_dict


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x
