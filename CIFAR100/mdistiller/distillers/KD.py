import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def fkl_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def rkl_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_teacher = F.log_softmax(logits_teacher / temperature, dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_teacher, pred_student, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def all_kl_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_teacher = F.log_softmax(logits_teacher / temperature, dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    fkl_loss = F.kl_div(log_pred_teacher, pred_student, reduction="none").sum(1).mean()

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    rkl_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()

    loss_kd = 0.5 * fkl_loss + 0.5 *rkl_loss
    loss_kd *= temperature**2
    return loss_kd

def get_ratio(teacher_logits, logits, mu=0.5):  
    # [B, L, V]  
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)  
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)  

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)  
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()  

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)  
    re_student_probs = student_probs.gather(dim=-1, index=idx)  

    errors = torch.abs(re_teacher_probs - re_student_probs)  

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)
    mask = cum_sum > mu  

    mask[:, 0] = False

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)  
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)  

    return s1 / (s1 + s2), s2 / (s1 + s2)

def akl_loss(logits_student_in, logits_teacher_in, temperature, logit_stand, mu=0.5):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_teacher = F.log_softmax(logits_teacher / temperature, dim=1)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    h_ratio, l_ratio = get_ratio(logits_teacher, logits_student, mu)

    fkl_loss = torch.sum(h_ratio.unsqueeze(1) * F.kl_div(log_pred_student, pred_teacher, reduction="none"), dim=-1).mean() 
    rkl_loss = torch.sum(l_ratio.unsqueeze(1) * F.kl_div(log_pred_teacher, pred_student, reduction="none"), dim=-1).mean() 
    loss_kd = fkl_loss + rkl_loss
    loss_kd *= temperature**2
    
    return loss_kd

def sfkl_loss(logits, teacher_logits, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    prod_probs = teacher_probs * mixed_logprobs
    
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.mean(x)
    
    return distil_loss

def srkl_loss(logits, teacher_logits, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)
    
    prod_probs = student_probs * mixed_logprobs - student_probs * student_logprobs
    
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.mean(x)
    
    return distil_loss

def bkl_loss(logits_student, logits_teacher, alpha, beta, temperature):
    log_pred_teacher = F.log_softmax(logits_teacher / temperature, dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    loss_fkd = F.kl_div(log_pred_student, pred_teacher, reduction="none")
    loss_rkd = F.kl_div(log_pred_teacher, pred_student, reduction="none")

    rho = ((pred_student ** 2) + (pred_teacher ** 2)) ** (0.5 * (1 - beta))

    epsilon = 1e-12
    loss_part1 = (
        (loss_fkd / (rho + epsilon))
    ).sum(1)
    loss_part2 = (
        (loss_rkd / (rho + epsilon))
    ).sum(1)

    loss_all = alpha * loss_part1 + (1 - alpha) * loss_part2

    loss_akd = (
        loss_all.sum(0)
        / logits_student.shape[0]
    )
    
    loss_akd *= temperature ** 2
    
    return loss_akd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.kl_type = cfg.KD.KL_TYPE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # losses
        if self.kl_type == 'fkl':
            loss_kd = self.kd_loss_weight * fkl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
        if self.kl_type == 'rkl':
            loss_kd = self.kd_loss_weight * rkl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
        if self.kl_type == 'all_kl':
            loss_kd = self.kd_loss_weight * (0.5 * fkl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            ) + 0.5 * rkl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            ))
        if self.kl_type == 'sfkl':
            loss_kd = self.kd_loss_weight * sfkl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
        if self.kl_type == 'srkl':
            loss_kd = self.kd_loss_weight * srkl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
        if self.kl_type == 'akl':
            loss_kd = self.kd_loss_weight * akl_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
        if self.kl_type == 'bkl':
            loss_kd = self.kd_loss_weight * bkl_loss(
                logits_student, logits_teacher, self.alpha, self.beta, self.temperature
            )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict