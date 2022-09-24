import logging

import torch

from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from typing import Type

logger = logging.getLogger(__name__)


def wrap_FedRepTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """
    Implementation of FedRep refer to `Exploiting Shared Representations for
    Personalized Federated Learning` [Liam Collins, et al., 2021]
        (https://arxiv.org/abs/2102.07078)

    """

    init_FedRep_ctx(base_trainer)

    base_trainer.register_hook_in_train(new_hook=hook_on_fit_start_fedrep,
                                        trigger="on_fit_start",
                                        insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=hook_on_epoch_start_fedrep,
                                        trigger="on_epoch_start",
                                        insert_pos=0)

    return base_trainer


def init_FedRep_ctx(base_trainer):
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.epoch_feature = cfg.fedrep.epoch_feature
    ctx.epoch_linear = cfg.fedrep.epoch_linear

    ctx.num_train_epoch = ctx.epoch_feature + ctx.epoch_linear

    ctx.cur_epoch_number = 0

    ctx.lr_feature = cfg.fedrep.lr_feature
    ctx.lr_linear = cfg.fedrep.lr_linear

    ctx.local_update_param = []
    ctx.global_update_param = []

    all_para_names = list(ctx.model.state_dict().keys())
    head_paras = all_para_names[-2:]  # e.g., linear.weight, linear.bias
    if type(base_trainer).LOG_META_INFO_TIME == 0:
        logger.info("For FedRep Trainer, we will merge the local head paras "
                    "into local personalization.local_param. Before merging,"
                    f"the last head para:\n head_paras={head_paras}\n"
                    f"the personalization.local_param="
                    f"{cfg.personalization.local_param}")
        type(base_trainer).LOG_META_INFO_TIME = 1

    for name in head_paras:
        if name not in cfg.personalization.local_param:
            cfg.personalization.local_param.append(name)

    for name, param in ctx.model.named_parameters():
        if name in ctx.cfg.personalization.local_param:
            ctx.local_update_param.append(param)
        else:
            ctx.global_update_param.append(param)

    ctx.optimizer_for_feature = torch.optim.SGD(ctx.global_update_param,
                                                lr=ctx.lr_feature)
    ctx.optimizer_for_linear = torch.optim.SGD(ctx.local_update_param,
                                               lr=ctx.lr_linear)


def hook_on_fit_start_fedrep(ctx):
    ctx.num_train_epoch = ctx.epoch_feature + ctx.epoch_linear
    ctx.cur_epoch_number = 0

    #  For the first ctx.epoch_linear epochs,
    #  only the linear classifier can be updated.

    for name, param in ctx.model.named_parameters():
        if name in ctx.cfg.personalization.local_param:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ctx.optimizer = ctx.optimizer_for_linear


def hook_on_epoch_start_fedrep(ctx):
    ctx.cur_epoch_number += 1

    if ctx.cur_epoch_number == ctx.epoch_linear + 1:

        for name, param in ctx.model.named_parameters():
            if name in ctx.cfg.personalization.local_param:
                param.requires_grad = False
            else:
                param.requires_grad = True

        ctx.optimizer = ctx.optimizer_for_feature
