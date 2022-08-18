import collections
import copy
import logging

from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_param_dict, merge_dict

import numpy as np

logger = logging.getLogger(__name__)


class HypClusterServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(HypClusterServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        self.msg_buffer['cluster'] = dict()
        self.cluster_models = dict()
        self.cluster_aggregators = [
            copy.deepcopy(self.aggregators) for _ in range(self.model_num - 1)
        ]

    def _register_default_handlers(self):
        super(HypClusterServer, self)._register_default_handlers()
        self.register_handlers('cluster', self.callback_funcs_for_cluster)

    def _init_cluster_models(self):
        msg_buffer = self.msg_buffer['train'][self.state]
        # choose q clients as the initialized model for each cluster
        clients = np.random.choice(list(msg_buffer.keys()),
                                   size=self._cfg.hypcluster.q,
                                   replace=False)
        for cluster_id, client_id in enumerate(clients):
            _, model_param = msg_buffer[client_id]
            self.cluster_models[cluster_id] = model_param

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):

        # The minimal number of clients
        minimal_number = self.client_num if check_eval_result \
            else self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):
            if not check_eval_result:
                # training stage
                if self.state == 0:
                    # Train P models in the first round
                    # random initialize the model for q clusters
                    self._init_cluster_models()
                else:
                    # aggregate model for each cluster
                    self._perform_federated_aggregation()

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    receiver = list(self.comm_manager.neighbors.keys())
                    self.comm_manager.send(
                        Message(msg_type='evaluate',
                                sender=self.ID,
                                receiver=receiver,
                                state=min(self.state, self.total_round_num),
                                timestamp=self.cur_timestamp,
                                content=self.cluster_models))

                if self.state < self.total_round_num:
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    if self.state > 1:
                        self.msg_buffer['cluster'][self.state - 1].clear()
                    self.staled_msg_buffer.clear()
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    logger.info('Begin to cluster clients '
                                'according to their performance')
                    # start a new cluster round
                    self._cluster_clients()
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # for evaluation
                self._merge_and_format_eval_results()

    def cluster_and_move_on(self):
        cluster_results = self.msg_buffer['cluster'][self.state]
        if sum([len(_)
                for _ in cluster_results.values()]) >= self.sample_client_num:
            # receive all cluster infos
            # start a new training round
            logger.info(f"The cluster results: "
                        f"{dict(self.msg_buffer['cluster'][self.state])}")
            # broadcast according to the cluster results
            for cluster_id, clients_id in cluster_results.items():
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=clients_id,
                            state=min(self.state, self.total_round_num),
                            timestamp=self.cur_timestamp,
                            content=self.cluster_models[cluster_id]))

            # self.broadcast_model_para(msg_type='model_para',
            #                           receiver=self.receiver)
            return True
        else:
            return False

    def _cluster_clients(self):
        receiver = self.sampler.sample(size=self.sample_client_num)
        # Send and evaluate all cluster models to clients
        self.comm_manager.send(
            Message(msg_type="cluster",
                    sender=self.ID,
                    receiver=receiver,
                    state=min(self.state, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=self.cluster_models))

    def callback_funcs_for_cluster(self, message: Message):
        round = message.state
        sender = message.sender
        cluster_id = message.content

        if round not in self.msg_buffer['cluster']:
            self.msg_buffer['cluster'][round] = collections.defaultdict(list)

        self.msg_buffer['cluster'][round][cluster_id].append(sender)

        return self.cluster_and_move_on()

    def _perform_federated_aggregation(self):
        msg_buffer = self.msg_buffer['train'][self.state]
        clusters = self.msg_buffer['cluster'][self.state]
        for cluster_id, clients_id in clusters.items():
            msg_list = [msg_buffer[_] for _ in clients_id]
            agg_info = {'client_feedback': msg_list}
            model_param = self.cluster_models[cluster_id]
            # TODO: check bug for un-seen clients
            result = self.aggregator.aggregate(agg_info)
            merged_param = merge_param_dict(model_param.copy(), result)
            self.cluster_models[cluster_id] = merged_param


class HypClusterClient(Client):
    def _register_default_handlers(self):
        super(HypClusterClient, self)._register_default_handlers()
        self.register_handlers('cluster', self.callback_funcs_for_cluster)

    def callback_funcs_for_cluster(self, message: Message):
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        content = message.content

        cluster_metrics = dict()
        for cluster_id, model_param in content.items():
            # update model parameters
            self.trainer.update(model_param, strict=False)

            eval_metrics = self.trainer.evaluate(
                target_data_split_name=self._cfg.hypcluster.split)

            cluster_metrics[cluster_id] = eval_metrics

        # TODO: support other metric
        metric = f'{self._cfg.hypcluster.split}_avg_loss'

        cluster_results = [
            cluster_metrics[key][metric] for key in cluster_metrics.keys()
        ]
        select_id = min(cluster_metrics.keys(),
                        key=lambda key: cluster_metrics[key][metric])

        logger.info(
            f"Client #{self.ID} is clustered into the "
            f"{select_id}-th cluster with the performances {cluster_results}.")

        # Return the evaluation results
        self.comm_manager.send(
            Message(msg_type='cluster',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=select_id))

    def callback_funcs_for_evaluate(self, message: Message):
        """
        Choose the clusters with the best performance

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        cluster_metrics = collections.defaultdict(dict)
        for cluster_id, model_param in message.content.items():
            self.trainer.update(model_param,
                                strict=self._cfg.federate.share_local_model)

            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                metrics.update(**eval_metrics)

            cluster_metrics[cluster_id].update(**metrics)

        # TODO: support more metrics
        # during evaluation, select cluster according to the validation loss
        select_id = min(cluster_metrics.keys(),
                        key=lambda key: cluster_metrics[key]['val_avg_loss'])
        metrics = cluster_metrics[select_id]

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))
