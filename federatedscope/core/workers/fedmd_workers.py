import logging

from federatedscope.core.workers.client import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)

# TODO: in fed_runner.py, add dataloader for the public dataset, whose key
#  is "public", and can be shared by all the clients in standalone mode


class FedMDClient(Client):
    """
        Client for "FedMD: Heterogenous Federated Learning via Model
        Distillation", https://arxiv.org/pdf/1910.03581.pdf

        In the implementation, we regard the transmitted "logits" as
        "model_para", to reuse the aggregation and message transition funcs
    """
    def callback_funcs_for_model_para(self, message: Message):
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        logits_of_server = message.content

        # at the beginning, do the pre-train, i.e., the "transfer learning"
        # stage in line 4 of the Algorithm 1 in FedMD paper
        if self.state == "0":
            self.trainer.ctx.train_mode = "pretrain_public"
            self.trainer.train(target_data_split_name="public")
            self.trainer.ctx.train_mode = "pretrain_private"
            self.trainer.train()

        # get logits on the public data
        self.trainer.ctx.model = self.trainer.ctx.model_for_public
        self.trainer.evaluate(target_data_split_name="public")
        self.trainer.ctx.model = self.trainer.ctx.model_for_private
        logits_of_client = self.trainer.ctx.logit2server

        # train on public data, the "digest" stage
        self.trainer.ctx.train_mode = "digest_public"
        self.trainer.ctx.logits_of_server = logits_of_server
        self.trainer.train(target_data_split_name="public")

        # train on private data, the "revisit" stage
        self.trainer.ctx.train_mode = "only_private"
        if self.early_stopper.early_stopped and \
                self._monitor.local_convergence_round == 0:
            logger.info(f"[Normal FL Mode] Client #{self.ID} has been locally "
                        f"early stopped. "
                        f"The next FL update may result in negative effect")
            self._monitor.local_converged()
        sample_size, model_para_all, results = self.trainer.train()
        train_log_res = self._monitor.format_eval_res(results,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True)
        logger.info(train_log_res)
        if self._cfg.wandb.use and self._cfg.wandb.client_train_info \
                and self._cfg.federate.client_num < 2000:
            self._monitor.save_formatted_results(train_log_res,
                                                 save_file_name="")

        # Return the evaluated logits
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(1, logits_of_client)
                    # "1" indicates uniform aggregation of the clients' logits
                    ))
