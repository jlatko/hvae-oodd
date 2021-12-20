import argparse
import logging
import os
from collections import defaultdict
import numpy as np
import pandas as pd

import rich
import torch
import wandb

from oodd.utils.oodd import compute_roc_pr_metrics
from oodd.utils.wandb import download_or_find

LOGGER = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument("--scores_run_id", type=str, help="wandb run id")
parser.add_argument("--complexities_run_id", type=str, help="wandb run id")
parser.add_argument("--run_name", type=str, default=None, help="name this wandb run")
parser.add_argument("--save_dir", type=str, default= "/scratch/s193223/oodd", help="directory for saving results")

args = parser.parse_args()
rich.print(vars(args))

SCORES_TO_SHOW = ["LLR", "ELBO", "LIKELIHOOD", "KL", "p_var_1", "p_var_sum"]


def load_data():
    path = download_or_find(args.scores_run_id, "all-scores.pt")
    data = torch.load(path)
    return data

def load_complexities():
    path = download_or_find(args.complexities_run_id, "complexity.pt")
    data = torch.load(path)
    return data

def setup_wandb():
    # api = wandb.Api()
    # run = api.run(f"johnnysummer/hvae/{run_id}")

    tags = ["scatter"]

    wandb.init(project="hvae", entity="johnnysummer", dir=args.save_dir, tags=tags)
    args.save_dir = wandb.run.dir
    wandb.config.update(args)
    #
    # if args.run_name is not None:
    #     run_name = args.run_name
    # elif run.name.split("-")[0] != "STATS_multiple":
    #     run_name = "_".join(
    #         "-".join(
    #             run.name.split("-")[:-1]
    #         ).split("_")[1:]
    #     )
    # else:
    #     run_name = str(run.name.split("-")[-1])
    #
    # run_name = "RESULTS_" + run_name + "-" + wandb.run.name.split("-")[-1]
    # wandb.run.name = run_name
    wandb.run.save()

    # wandb.save("*.csv")


# def plot_compression(comp="complexity_mean_local_entropy_3",
#                      k=0, key="elbos_k", data=elbo_k, size=4, logscale_y=False, neg=False, logscale_x=False):
#     colors = "rgbcmyk"
#     dataset_pairs = [
#         ('CIFAR10 test', 'SVHN test'),
#         ('FashionMNIST test', 'MNIST test'),
#     ]
#     fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(4 * size, 2 * size), facecolor="w")
#     for i, datasets in enumerate(dataset_pairs):
#         for j, reference_dataset in enumerate(datasets):
#             ax = axes[j][i]
#             for l, target_dataset in enumerate(dataset_groups[reference_dataset]):
#                 c = colors[2 * i + l]
#                 print(reference_dataset, target_dataset)
#                 print(len(data[reference_dataset][target_dataset][key][k][1][1]))
#                 print(len(complexities[comp][target_dataset]))
#                 values = data[reference_dataset][target_dataset][key][k][1][1][:10000]
#                 if neg:
#                     values = -values
#                 if logscale_y:
#                     values = np.log(values)
#                 comps = complexities[comp][target_dataset][:10000]
#                 r = pearsonr(values, comps)
#                 print(reference_dataset, target_dataset, r)
#
#                 if logscale_x:
#                     comps = np.log(comps)
#
#                 ax.scatter(comps, values, alpha=0.01, color=c)
#                 z = np.polyfit(comps, values, 1)
#                 p = np.poly1d(z)
#                 ax.plot(comps, p(comps), label=target_dataset.split()[0], color=c)
#
#             ax.legend()
#             ylabel = f"{key} ({reference_dataset.split()[0]} model)"
#
#             if neg:
#                 ylabel = "neg " + ylabel
#             if logscale_y:
#                 ylabel = "log  " + ylabel
#
#             ax.set_ylabel(ylabel)
#
#     xlabel = comp
#     if logscale_x:
#         xlabel = "log  " + xlabel
#     plt.xlabel(xlabel)
#     plt.subplots_adjust(hspace=.0)

if __name__ == "__main__":
    ALL_RESULTS = []
    setup_wandb()
    data = load_data()
    complexities = load_complexities()

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))))
    reference_datasets = sorted(
        list(data.keys()),
        key=lambda x: ("a" if "Binarized" in x else "b") + x.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
    )
    for reference_dataset in reference_datasets:
        reference_dataset_stripped = reference_dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
        test_datasets = sorted(list(data[reference_dataset].keys()))

        # check if ok
        reference_dataset_key = [test_dataset for test_dataset in test_datasets if test_dataset.split()[0] == reference_dataset_stripped]
        if len(reference_dataset_key) == 0:
            print(f"ERROR: {reference_dataset_stripped} not in {test_datasets}")
            continue
        if len(reference_dataset_key) > 1:
            print(f"WARNING: {reference_dataset_stripped} appears more then once in: {test_datasets}")

        reference_dataset_key = reference_dataset_key[0]

        # print(f"========== {reference_dataset} (in-distribution) ==========\n")
        all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        all_complexities = []
        for test_dataset in test_datasets:
            if test_dataset not in complexities:
                print(f"WARNING: {test_dataset} not in complexities")
                continue

            all_complexities.append(complexities[test_dataset])

            k_values = sorted(list(data[reference_dataset][test_dataset].keys()))
            for k in k_values:
                run_ids = sorted(list(data[reference_dataset][test_dataset][k].keys()))
                for run_id in run_ids:
                    s = f"({run_id}) {k} | "
                    score_names = sorted(list(data[reference_dataset][test_dataset][k][run_id].keys()))
                    for score_name in score_names:
                        if score_name in SCORES_TO_SHOW:
                            test_scores = np.array(data[reference_dataset][test_dataset][k][run_id][score_name])
                            any_nan = False
                            if np.any(np.isnan(test_scores)):
                                if score_name in SCORES_TO_SHOW:
                                    print(f"WARNING: nan in test {score_name}")
                                test_scores = np.nan_to_num(test_scores, copy=True, nan=0.0, posinf=1e10, neginf=-1e10)
                                any_nan = True

                            all_scores[k][run_id][score_name].append(test_scores)


