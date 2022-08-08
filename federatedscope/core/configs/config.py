import copy
import logging
import os
from collections import defaultdict

from pathlib import Path

import federatedscope.register as register
from federatedscope.core.configs.yacs_config import CfgNode, _merge_a_into_b, \
    Argument

logger = logging.getLogger(__name__)


def set_help_info(cn_node, help_info_dict, prefix=""):
    for k, v in cn_node.items():
        if isinstance(v, Argument) and k not in help_info_dict:
            help_info_dict[prefix + k] = v.description
        elif isinstance(v, CN):
            set_help_info(v,
                          help_info_dict,
                          prefix=f"{k}." if prefix == "" else f"{prefix}{k}.")


class CN(CfgNode):
    """
        An extended configuration system based on [yacs](
        https://github.com/rbgirshick/yacs).
        The two-level tree structure consists of several internal dict-like
        containers to allow simple key-value access and management.

    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        init_dict = super().__init__(init_dict, key_list, new_allowed)
        self.cfg_check_funcs = list()  # to check the config values validity
        self.help_info = defaultdict(str)  # build the help dict

        if init_dict:
            for k, v in init_dict.items():
                if isinstance(v, Argument):
                    self.help_info[k] = v.description
                elif isinstance(v, CN):
                    for name, des in k.help_info:
                        self.help_info[name] = des

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def print_help(self, arg_name=""):
        """
            print help info for a specific given `arg_name` or
            for all arguments if not given `arg_name`
        :param arg_name:
        :return:
        """
        if arg_name != "" and arg_name in self.help_info:
            print(f"  --{arg_name} \t {self.help_info[arg_name]}")
        else:
            for k, v in self.help_info.items():
                print(f"  --{k} \t {v}")

    def register_cfg_check_fun(self, cfg_check_fun):
        self.cfg_check_funcs.append(cfg_check_fun)

    def merge_from_file(self, cfg_filename):
        """
            load configs from a yaml file, another cfg instance or a list
            stores the keys and values.

        :param cfg_filename (string):
        :return:
        """
        cfg_check_funcs = copy.copy(self.cfg_check_funcs)
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        self.merge_from_other_cfg(cfg)
        self.cfg_check_funcs.clear()
        self.cfg_check_funcs.extend(cfg_check_funcs)
        self.assert_cfg()
        set_help_info(self, self.help_info)

    def merge_from_other_cfg(self, cfg_other):
        """
            load configs from another cfg instance

        :param cfg_other (CN):
        :return:
        """

        cfg_check_funcs = copy.copy(self.cfg_check_funcs)
        _merge_a_into_b(cfg_other, self, self, [])
        self.cfg_check_funcs.clear()
        self.cfg_check_funcs.extend(cfg_check_funcs)
        self.assert_cfg()
        set_help_info(self, self.help_info)

    def merge_from_list(self, cfg_list):
        """
           load configs from a list stores the keys and values.
           modified `merge_from_list` in `yacs.config.py` to allow adding
           new keys if `is_new_allowed()` returns True

        :param cfg_list (list):
        :return:
        """
        cfg_check_funcs = copy.copy(self.cfg_check_funcs)
        super().merge_from_list(cfg_list)
        self.cfg_check_funcs.clear()
        self.cfg_check_funcs.extend(cfg_check_funcs)
        self.assert_cfg()
        set_help_info(self, self.help_info)

    def assert_cfg(self):
        """
            check the validness of the configuration instance

        :return:
        """
        for check_func in self.cfg_check_funcs:
            check_func(self)

    def clean_unused_sub_cfgs(self):
        """
            Clean the un-used secondary-level CfgNode, whose `.use`
            attribute is `True`

        :return:
        """
        for v in self.values():
            if isinstance(v, CfgNode) or isinstance(v, CN):
                # sub-config
                if hasattr(v, "use") and v.use is False:
                    for k in copy.deepcopy(v).keys():
                        # delete the un-used attributes
                        if k == "use":
                            continue
                        else:
                            del v[k]

    def _check_required_args(self):
        for k, v in self.items():
            if isinstance(v, CN):
                v._check_required_args()
            if isinstance(v, Argument) and v.required and v.value is None:
                logger.warning(f"You have not set the required argument {k}")

    def freeze(self, inform=True, save=True):
        """
            1) make the cfg attributes immutable;
            2) save the frozen cfg_check_funcs into
            "self.outdir/config.yaml" for better reproducibility;
            3) if self.wandb.use=True, update the frozen config

        :return:
        """
        self.assert_cfg()
        self.clean_unused_sub_cfgs()
        self._check_required_args()
        if save:  # save the final cfg
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.outdir, "config.yaml"),
                      'w') as outfile:
                from contextlib import redirect_stdout
                with redirect_stdout(outfile):
                    tmp_cfg = copy.deepcopy(self)
                    tmp_cfg.cfg_check_funcs.clear()
                    tmp_cfg.help_info.clear()
                    print(tmp_cfg.dump())
                if self.wandb.use:
                    # update the frozen config
                    try:
                        import wandb
                        import yaml
                        cfg_yaml = yaml.safe_load(tmp_cfg.dump())
                        wandb.config.update(cfg_yaml, allow_val_change=True)
                    except ImportError:
                        logger.error(
                            "cfg.wandb.use=True but not install the wandb "
                            "package")
                        exit()

            if inform:
                logger.info("the used configs are: \n" + str(tmp_cfg))

        super(CN, self).freeze()


# to ensure the sub-configs registered before set up the global config
from federatedscope.core.configs import all_sub_configs

for sub_config in all_sub_configs:
    __import__("federatedscope.core.configs." + sub_config)

from federatedscope.contrib.configs import all_sub_configs_contrib

for sub_config in all_sub_configs_contrib:
    __import__("federatedscope.contrib.configs." + sub_config)

# Global config object
global_cfg = CN()


def init_global_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''

    # ---------------------------------------------------------------------- #
    # Basic options, first level configs
    # ---------------------------------------------------------------------- #

    cfg.backend = 'torch'

    # Whether to use GPU
    cfg.use_gpu = False

    # Whether to print verbose logging info
    cfg.verbose = 1

    # How many decimal places we print out using logger
    cfg.print_decimal_digits = 6

    # Specify the device
    cfg.device = -1

    # Random seed
    cfg.seed = 0

    # Path of configuration file
    cfg.cfg_file = ''

    # The dir used to save log, exp_config, models, etc,.
    cfg.outdir = 'exp'
    cfg.expname = ''  # detailed exp name to distinguish different sub-exp
    cfg.expname_tag = ''  # detailed exp tag to distinguish different
    # sub-exp with the same expname

    # extend user customized configs
    for func in register.config_dict.values():
        func(cfg)

    set_help_info(cfg, cfg.help_info)


init_global_cfg(global_cfg)
