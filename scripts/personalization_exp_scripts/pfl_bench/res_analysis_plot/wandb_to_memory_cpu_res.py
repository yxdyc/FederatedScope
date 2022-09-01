import copy
import json
import os

import wandb
from collections import OrderedDict

import yaml

from scripts.personalization_exp_scripts.pfl_bench.res_analysis_plot.render_paper_res import \
    highlight_tex_res_in_table

api = wandb.Api()

filters_each_line_memory_cpu_table = OrderedDict(
    # {dataset_name: filter}
    [
        # ("all",
        # None,
        # ),
        # ("FEMNIST-all",
        #  {"$and":
        #      [
        #          {"config.data.type": "femnist"},
        #      ]
        #  }
        #  ),
        (
            "FEMNIST-s02",
            {
                "$and": [
                    {
                        "config.data.type": "femnist"
                    },
                    {
                        "config.federate.sample_client_rate": 0.2
                    },
                    # {"state": "finished"},
                ]
            }),
        # ("cifar10-alpha05",
        #  {"$and":
        #      [
        #          {"config.data.type": "CIFAR10@torchvision"},
        #          {"config.data.splitter_args": [{"alpha": 0.5}]},
        #      ]
        #  }
        #  ),
        ("sst2", {
            "$and": [
                {
                    "config.data.type": "sst2@huggingface_datasets"
                },
            ]
        }),
        # ("pubmed",
        # {"$and":
        #     [
        #         {"config.data.type": "pubmed"},
        #     ]
        # }
        # ),
    ])

sweep_name_2_id = dict()

column_names_sys = [
    "memory_mean",
    "memory_peak",
    #"cpu_mean",
    #"cpu_peak",
    "runtime"
]
sorted_method_name_pair = [
    ("global-train", "Global-Train"),
    ("isolated-train", "Isolated"),
    ("fedavg", "FedAvg"),
    ("fedavg-ft", "FedAvg-FT"),
    ("fedprox", "FedProx"),
    ("fedprox-ft", "FedProx-FT"),
    # ("fedopt", "FedOpt"),
    # ("fedopt-ft", "FedOpt-FT"),
    ("pfedme", "pFedMe"),
    ("ft-pfedme", "pFedMe-FT"),
    ("hypcluster", "HypCluster"),
    ("ft-hypcluster", "HypCluster-FT"),
    ("fedbn", "FedBN"),
    ("fedbn-ft", "FedBN-FT"),
    ("fedbn-fedopt", "FedBN-FedOPT"),
    ("fedbn-fedopt-ft", "FedBN-FedOPT-FT"),
    ("ditto", "Ditto"),
    ("ditto-ft", "Ditto-FT"),
    ("ditto-fedbn", "Ditto-FedBN"),
    ("ditto-fedbn-ft", "Ditto-FedBN-FT"),
    ("ditto-fedbn-fedopt", "Ditto-FedBN-FedOpt"),
    ("ditto-fedbn-fedopt-ft", "Ditto-FedBN-FedOpt-FT"),
    ("fedem", "FedEM"),
    ("fedem-ft", "FedEM-FT"),
    ("fedbn-fedem", "FedEM-FedBN"),
    ("fedbn-fedem-ft", "FedEM-FedBN-FT"),
    ("fedbn-fedem-fedopt", "FedEM-FedBN-FedOPT"),
    ("fedbn-fedem-fedopt-ft", "FedEM-FedBN-FedOPT-FT"),
]

sorted_method_name_to_print = OrderedDict(sorted_method_name_pair)
expected_keys = set(list(sorted_method_name_to_print.keys()))
expected_method_names = list(sorted_method_name_to_print.values())

expected_datasets_name = [
    "FEMNIST-s02",
    "sst2",
]

expected_seed_set = ["1", "2", "3"]
expected_expname_tag = set()
found_expname_tag = set()

for method_name in expected_method_names:
    for dataset_name in expected_datasets_name:
        for seed in expected_seed_set:
            expected_expname_tag.add(f"{method_name}_{dataset_name}")
from collections import defaultdict

all_res_structed = defaultdict(dict)
for expname_tag in expected_expname_tag:
    for metric in column_names_sys:
        all_res_structed[expname_tag][metric] = "-"


def load_print_cpu_memory_res(highlight_tex=True):
    filtered_runs = api.runs("pfl-bench-best-repeat")
    for run in filtered_runs:
        expname_tag = run.config["expname_tag"]
        method, dataname, seed = expname_tag.split("_")
        expname_no_seed = f"{method}_{dataname}"
        if expname_no_seed not in found_expname_tag and \
                expname_no_seed in expected_expname_tag:
            #if run.state == "failed":
            #    continue
            if run.state != "finished":
                print(f"run {run} is not fished, with name {expname_tag}")
            found_expname_tag.add(expname_no_seed)
            system_metrics = run.history(stream="events")
            all_res_structed[expname_no_seed]["memory_mean"] = \
                system_metrics["system.proc.memory.rssMB"].mean()
            all_res_structed[expname_no_seed]["memory_peak"] = \
                system_metrics["system.proc.memory.rssMB"].max()
            # all_res_structed[expname_no_seed]["cpu_mean"] = \
            #     system_metrics["system.proc.cpu.threads"].mean()
            # all_res_structed[expname_no_seed]["cpu_peak"] = \
            #     system_metrics["system.proc.cpu.threads"].max()
            if '_runtime' in run.summary:
                time_run = run.summary["_runtime"]
            else:
                time_run = run.summary['_wandb']["runtime"]
            all_res_structed[expname_no_seed]["runtime"] = time_run

    if len(found_expname_tag) != len(expected_expname_tag):
        print(f"Missing, found {len(found_expname_tag)}, but expected"
              f" {len(expected_expname_tag)}\n")
        print(f"The missing ones are"
              f"{expected_expname_tag - found_expname_tag}")

    print("\n=============res_of_each_line [memory,cpu]===============" +
          ",".join(list(filters_each_line_memory_cpu_table.keys())))

    # memory mean,peak, cpu mean, peak
    res_to_print_matrix = []
    try:
        for method_name in expected_method_names:
            res_to_print = [method_name]
            for dataset_name in expected_datasets_name:
                exp_name_tag = f"{method_name}_{dataset_name}"
                res_to_print_per_data = [
                    "{:.2f}".format(
                        all_res_structed[exp_name_tag][metric_name])
                    for metric_name in column_names_sys
                ]
                res_to_print.extend(res_to_print_per_data)
            # print(",".join(res_to_print))
            res_to_print_matrix.append(res_to_print)
    except Exception as e:
        print(f"exception {e}")

    if highlight_tex:
        colum_order_per_data = ["-"] * len(column_names_sys)
        rank_order = colum_order_per_data * \
                     len(filters_each_line_memory_cpu_table)
        res_to_print_matrix = highlight_tex_res_in_table(res_to_print_matrix,
                                                         rank_order=rank_order)
        for res_to_print in res_to_print_matrix:
            print("&".join(res_to_print) + "\\\\")
    else:
        for res_to_print in res_to_print_matrix:
            print("\t".join(res_to_print))


load_print_cpu_memory_res(highlight_tex=False)
