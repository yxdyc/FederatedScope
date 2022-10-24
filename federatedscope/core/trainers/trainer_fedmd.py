import copy
import logging

import torch

from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from typing import Type

logger = logging.getLogger(__name__)


def wrap_FedMDTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """
    Implementation of FedMD refer to `FedMD: Heterogenous Federated Learning
        via Model Distillation` [Daliang Li, Junpu Wang, 2019]
        (https://arxiv.org/abs/1910.03581)

    """

    _init_FedMD_ctx(base_trainer)

    base_trainer.register_hook_in_train(new_hook=_hook_on_fit_start_FedMD,
                                        trigger="on_fit_start",
                                        insert_pos=-1)
    base_trainer.register_hook_in_train(
        new_hook=_hook_on_batch_forward_digest_public,
        trigger="on_batch_forward",
        insert_pos=-1)

    # to get the logits on public data
    base_trainer.register_hook_in_eval(new_hook=_hook_on_fit_start_FedMD,
                                       trigger="on_fit_start",
                                       insert_pos=-1)
    base_trainer.register_hook_in_eval(
        new_hook=_hook_on_batch_forward_track_logits,
        trigger="on_batch_forward",
        insert_pos=-1)

    return base_trainer


def _init_FedMD_ctx(base_trainer):
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.epoch_pretrain_public = cfg.fedmd.epoch_pretrain_public
    ctx.epoch_pretrain_private = cfg.fedmd.epoch_pretrain_private
    ctx.epoch_digest = cfg.fedmd.epoch_digest_public
    ctx.epoch_private = ctx.num_train_epoch

    ctx.model_for_private = ctx.model
    ctx.model_for_public = copy.copy(ctx.model)
    ctx.train_mode = [
        "pretrain_public", "pretrain_private", "digest", "only_private"
    ][3]

    # the L1 loss is according to the implementioan at
    # https://github.com/Koukyosyumei/AIJack/blob/main/src/
    # aijack/collaborative/fedmd/client.py#L17
    ctx.kd_loss_func = torch.nn.L1Loss()

    # init the model for public dataset, which has a distinct last layer
    # as it may have output shape different from the model for private data
    all_para_pairs = list(ctx.model_for_public.state_dict())
    # the last two are usually: linear.weight, linear.bias
    last_hidden_shape = all_para_pairs[-2][-1].shape[-1]
    out_shape_public = cfg.fedmd.public_out_channels
    last_module_key = list(ctx.model_for_public._modules.keys())[-1]
    ctx.model_for_public._modules[last_module_key] \
        = torch.nn.Linear(last_hidden_shape, out_shape_public)

    ctx.optimizer_for_private = ctx.optimizer
    ctx.optimizer_for_public = torch.optim.SGD(
        ctx.model_for_public.parameters(), lr=cfg.fedmd.lr_public)

    # logits on public data
    ctx.logit2server = torch.ones(
        (ctx.num_public_batch, out_shape_public)).to(ctx.device) * float("inf")
    ctx.cur_data_batch_idx = 0


def _hook_on_fit_start_FedMD(ctx):
    ctx.optimizer = ctx.optimizer_for_private
    ctx.model = ctx.model_for_private

    if ctx.train_mode == "pretrain_public":
        ctx.num_train_epoch = ctx.epoch_pretrain_public
        ctx.optimizer = ctx.optimizer_for_public
        ctx.model = ctx.model_for_public
    elif ctx.train_mode == "pretrain_private":
        ctx.num_train_epoch = ctx.epoch_pretrain_private
    elif ctx.train_mode == "digest":
        ctx.num_train_epoch = ctx.epoch_digest
    elif ctx.train_mode == "only_private":
        ctx.num_train_epoch = ctx.epoch_private
    else:
        raise ValueError(f"ctx.train_mode must be one of [`pretrain_public`, "
                         f"`pretrain_private`, `digest`, `only_private`]."
                         f"but got {ctx.train_mode}")


def _hook_on_batch_forward_digest_public(ctx):
    if ctx.train_mode == "digest":
        ctx.loss_batch = ctx.kd_loss_func(
            ctx.y_prob, ctx.logits_of_server[ctx.cur_data_batch_idx, :])
        # For the dataloader of public data, we assume its Shuffle=False
        ctx.cur_data_batch_idx = \
            (ctx.cur_data_batch_idx + 1) % ctx.num_public_batch


def _hook_on_batch_forward_track_logits(ctx):
    ctx.logit2server[ctx.cur_data_batch_idx, :] = ctx.y_prob.detach()
    # For the dataloader of public data, we assume its Shuffle=False
    ctx.cur_data_batch_idx = \
        (ctx.cur_data_batch_idx + 1) % ctx.num_public_batch
