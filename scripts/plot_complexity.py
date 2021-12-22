import argparse
import logging
import os
from collections import defaultdict
import numpy as np
import pandas as pd

import rich
import torch
import wandb
from scipy.stats import pearsonr

from oodd.utils.oodd import compute_roc_pr_metrics
from oodd.utils.wandb import download_or_find
import matplotlib.pyplot as plt

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
    print("loading: ", path)
    data = torch.load(path)
    return data

def load_complexities():
    path = download_or_find(args.complexities_run_id, "complexity.pt")
    print("loading: ", path)
    data = torch.load(path)
    return data

def setup_wandb():
    # api = wandb.Api()
    # run = api.run(f"johnnysummer/hvae/{run_id}")
    # TODO add tags/meta from both
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

#
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

def plot_compression(datasets, complexities, scores, title):
    plt.figure(figsize=(7,7))
    plt.title(title)
    colors = "rgbcmyk"

    for l, dataset in enumerate(datasets):
        print(title, dataset, len(scores[dataset]), len(complexities[dataset]))
        c = colors[l]

        comps = complexities[dataset][:10000]
        values = scores[dataset][:10000]
        if len(comps) != len(values):
            print(f"WARNING: {len(comps)} != {len(values)} (comp vs scores) in {dataset} ({title})")
            continue

        r = pearsonr(values, comps)
        print(reference_dataset, dataset, r)

        plt.scatter(comps, values, alpha=0.01, color=c)
        z = np.polyfit(comps, values, 1)
        p = np.poly1d(z)
        plt.plot(comps, p(comps), label=dataset.split()[0], color=c)
    plt.legend()


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
    print(list(complexities.keys()))
    for reference_dataset in reference_datasets:
        # reference_dataset_stripped = reference_dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
        test_datasets = sorted(list(data[reference_dataset].keys()))

        # check if ok
        reference_dataset_key = [test_dataset for test_dataset in test_datasets if test_dataset.split()[0] == reference_dataset]
        if len(reference_dataset_key) == 0:
            print(f"ERROR: {reference_dataset} not in {test_datasets}")
            continue
        if len(reference_dataset_key) > 1:
            print(f"WARNING: {reference_dataset} appears more then once in: {test_datasets}")

        reference_dataset_key = reference_dataset_key[0]

        # print(f"========== {reference_dataset} (in-distribution) ==========\n")
        all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for test_dataset in test_datasets:
            if test_dataset not in complexities:
                print(f"WARNING: {test_dataset} not in complexities")
                continue

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

                            all_scores[k][run_id][score_name][test_dataset] = test_scores

        # plot
        k_values = sorted(list(all_scores.keys()))
        for k in k_values:
            run_ids = sorted(list(all_scores[k].keys()))
            for run_id in run_ids:
                score_names = sorted(list(all_scores[k][run_id].keys()))
                for score_name in score_names:
                    test_datasets = sorted(list(all_scores[k][run_id][score_name].keys()))

                    try:
                        plot_compression(test_datasets,
                                         complexities=complexities,
                                         scores=all_scores[k][run_id][score_name],
                                         title=f"{k} {score_name}")
                        name = f"{reference_dataset} ({run_id}) {k} {score_name}"
                        wandb.log({name + "img": wandb.Image(plt)})

                        plt.savefig(os.path.join(wandb.run.dir, f"{reference_dataset}_{run_id}_{k}_{score_name}"))
                        # wandb.log({name: plt})
                    except Exception as e:
                        print("Caught exception for:", reference_dataset, run_id, k, score_name)
                        print(e)

                    # plt.savefig()








