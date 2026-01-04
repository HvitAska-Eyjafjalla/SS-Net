import logging
import logging.handlers

import os
import random
import numpy as np
import pandas as pd
import torch

from thop import profile
# from torchstat import stat
import torch.nn.functional as F

import torch.nn as nn
import math

# from torch.optim.lr_scheduler import LRScheduler
from collections import deque


class MyLogger:
    def __init__(self, logger_name, log_directory, universal_config, model_config):
        self.logger_name = logger_name
        self.log_directory = log_directory
        self.universal_config = universal_config
        self.model_config = model_config

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)

    def creat_info_file(self):
        info_name = self.log_directory / f'{self.logger_name}.info.log'
        info_handler = logging.handlers.TimedRotatingFileHandler(str(info_name), when='D', encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        info_handler.setFormatter(formatter)

        self.logger.addHandler(info_handler)

    def log_UniversalConfig_info(self):
        config_dict = self.universal_config.__dict__
        log_info = f'#----------Universal Config info----------#'
        self.logger.info(log_info)
        for k, v in config_dict.items():
            if k[0] == '_':
                continue
            else:
                log_info = f'{k}: {v},'
                self.logger.info(log_info)

    def log_ModelConfig_info(self):
        config_dict = self.model_config.__dict__
        log_info = f'#----------Model Config info----------#'
        self.logger.info(log_info)
        for k, v in config_dict.items():
            if k[0] == '_':
                continue
            else:
                log_info = f'{k}: {v},'
                self.logger.info(log_info)

    def log_and_print_custom_info(self, info, indent=False):
        self.logger.info(info)
        if indent:
            print('\t'+info)
        else:
            print(info)



def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    # cudnn.benchmark = True
    # cudnn.deterministic = True


def calculating_params_flops(model, size, Mylogger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print("\tFLOPs: %.4fG" % (flops / 1e9))
    print("\tParams: %.4fM" % (params / 1e6))

    total = sum(p.numel() for p in model.parameters())
    print("\tTotal params: %.4fM" % (total / 1e6))

    Mylogger.logger.info('#----------Model info----------#')
    Mylogger.logger.info(f'Flops: {flops / 1e9:.4f}G, Params: {params / 1e6:.4f}M, Total params: : {total / 1e6:.4f}M')


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6], n_labels=1):
        super(WeightedBCE, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        if self.n_labels == 1:
            logit = logit_pixel.view(-1).float()
            truth = truth_pixel.view(-1)
            assert (logit.shape == truth.shape)
            loss = F.binary_cross_entropy(logit, truth, reduction='none')
            # loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()

            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

            return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5], n_labels=1):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit, truth, smooth=1e-5):
        if (self.n_labels == 1):
            batch_size = len(logit)
            logit = logit.reshape(batch_size, -1)
            truth = truth.reshape(batch_size, -1)
            assert (logit.shape == truth.shape)

            # logit = torch.sigmoid(logit)

            p = logit.reshape(batch_size, -1)
            t = truth.reshape(batch_size, -1)
            w = truth.detach()
            w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
            # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
            # t = w*(t*2-1)
            p = w * p
            t = w * t
            intersection = (p * t).sum(-1)
            union = (p * p).sum(-1) + (t * t).sum(-1)
            dice = 1 - (2 * intersection + smooth) / (union + smooth)
            # print "------",dice.data

            loss = dice.mean()
            return loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1, n_labels=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5], n_labels=n_labels)
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5], n_labels=n_labels)
        self.n_labels = n_labels
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


def get_optimizer(config, model):
    assert config.optimizer in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                                'SGD'], 'Unsupported optimizer!'

    if config.optimizer == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.optimizer == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.optimizer == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.optimizer == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.scheduler in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                                'CosineAnnealingWarmRestarts', 'WP_MultiStepLR',
                                'WP_CosineLR', 'AdaptiveLinearAnnealingSoftRestarts'], 'Unsupported scheduler!'
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'WP_MultiStepLR':
        lr_func = lambda \
                epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.scheduler == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.total_epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler
