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
from oodd.utils.wandb import download_or_find
from oodd.constants import WANDB_USER, WANDB_PROJECT, DATA_PATH

LOGGER = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="wandb run id")
parser.add_argument("--run_name", type=str, default=None, help="name this wandb run")
parser.add_argument("--save_dir", type=str, default= f"{DATA_PATH}/oodd", help="directory for saving results")

args = parser.parse_args()
rich.print(vars(args))


SCORES_TO_NEGATE = [
    "ELBO", 'LIKELIHOOD',
    "ELBO A", 'LIKELIHOOD A',
    "ELBO B", 'LIKELIHOOD B',
]
SCORES_TO_SHOW = ["LLR", "ELBO", "LIKELIHOOD", "KL", "p_var_1", "p_var_sum"]


def load_data(run_id):
    path = download_or_find(run_id, "all-scores.pt")
    data = torch.load(path)
    return data


def setup_wandb(run_id):
    api = wandb.Api()
    run = api.run(f"{WANDB_USER}/{WANDB_PROJECT}/{run_id}")

    tags = ["results", run_id]

    wandb.init(project=WANDB_PROJECT, entity=WANDB_USER, dir=args.save_dir, tags=tags)
    args.save_dir = wandb.run.dir
    wandb.config.update(args)

    if args.run_name is not None:
        run_name = args.run_name
    elif run.name.split("-")[0] != "STATS_multiple":
        run_name = "_".join(
            "-".join(
                run.name.split("-")[:-1]
            ).split("_")[1:]
        )
    else:
        run_name = str(run.name.split("-")[-1])

    run_name = "RESULTS_" + run_name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.run.save()

    wandb.save("*.csv")

if __name__ == "__main__":
    ALL_RESULTS = []
    setup_wandb(args.run_id)
    data = load_data(args.run_id)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))))
    reference_datasets = sorted(
        list(data.keys()),
        key=lambda x: ("a" if "Binarized" in x else "b") + x.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
    )
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

        # TODO: is it reference?
        reference_dataset_key = reference_dataset_key[0]

        # print(f"========== {reference_dataset} (in-distribution) ==========\n")
        all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        ref_scores = defaultdict(lambda: defaultdict(dict))
        for test_dataset in test_datasets:
            if test_dataset.split()[0] == reference_dataset:
                continue

            print(f"--- {reference_dataset}  vs {test_dataset} ---")

            k_values = sorted(list(data[reference_dataset][test_dataset].keys()))
            for k in k_values:

                run_ids = sorted(list(data[reference_dataset][test_dataset][k].keys()))
                for run_id in run_ids:
                    s = f"({run_id}) {k} | "
                    score_names = sorted(list(data[reference_dataset][test_dataset][k][run_id].keys()))
                    for score_name in score_names:
                        reference_scores = np.array(data[reference_dataset][reference_dataset_key][k][run_id][score_name])
                        test_scores = np.array(data[reference_dataset][test_dataset][k][run_id][score_name])
                        any_nan = False

                        if np.any(np.isnan(reference_scores)):
                            if score_name in SCORES_TO_SHOW:
                                print(f"WARNING: nan in reference {score_name}")
                            reference_scores = np.nan_to_num(reference_scores, copy=True, nan=0.0, posinf=1e10, neginf=-1e10)
                            any_nan = True
                        if np.any(np.isnan(test_scores)):
                            if score_name in SCORES_TO_SHOW:
                                print(f"WARNING: nan in test {score_name}")
                            test_scores = np.nan_to_num(test_scores, copy=True, nan=0.0, posinf=1e10, neginf=-1e10)
                            any_nan = True

                        if score_name in SCORES_TO_NEGATE:
                            reference_scores = -reference_scores
                            test_scores = -test_scores

                        all_scores[k][run_id][score_name].append(test_scores)
                        ref_scores[k][run_id][score_name] = reference_scores

                        # compute metrics
                        y_true = np.array([*[0] * len(reference_scores), *[1] * len(test_scores)])
                        y_score = np.concatenate([reference_scores, test_scores])

                        (
                            (roc_auc, fpr, tpr, thresholds),
                            (pr_auc, precision, recall, thresholds),
                            fpr80,
                        ) = compute_roc_pr_metrics(y_true=y_true, y_score=y_score, reference_class=0)

                        results[reference_dataset][test_dataset][k][run_id][score_name] = dict(
                            roc=dict(roc_auc=roc_auc, fpr=fpr, tpr=tpr, thresholds=thresholds),
                            pr=dict(pr_auc=pr_auc, precision=precision, recall=recall, thresholds=thresholds),
                            fpr80=fpr80,
                        )

                        ALL_RESULTS.append({
                            "reference_dataset": reference_dataset,
                            "dataset": test_dataset,
                            "score_name": score_name,
                            "k": k,
                            # "iw_elbo": iw_elbo, ??
                            # "iw_elbo_k": iw_elbo_k, ??
                            "AUROC": roc_auc,
                            "AUPRC": pr_auc,
                            "FPR80": fpr80,
                            "stat": score_name,
                            "nan": any_nan,
                            "run_id": run_id
                        })

                        if score_name in SCORES_TO_SHOW:
                            s += f"{score_name:10s}: {roc_auc:.3f} | "

                    print(s)

        # ALL SCORES
        k_values = sorted(list(ref_scores.keys()))
        print(f"--- {reference_dataset}  vs ALL ---")
        for k in k_values:
            run_ids = sorted(list(ref_scores[k].keys()))
            for run_id in run_ids:
                s = f"({run_id}) {k} | "
                score_names = sorted(list(ref_scores[k][run_id].keys()))
                for score_name in score_names:
                    reference_scores = ref_scores[k][run_id][score_name]
                    test_scores = np.concatenate(all_scores[k][run_id][score_name])

                    # compute metrics
                    y_true = np.array([*[0] * len(reference_scores), *[1] * len(test_scores)])
                    y_score = np.concatenate([reference_scores, test_scores])

                    (
                        (roc_auc, fpr, tpr, thresholds),
                        (pr_auc, precision, recall, thresholds),
                        fpr80,
                    ) = compute_roc_pr_metrics(y_true=y_true, y_score=y_score, reference_class=0)

                    ALL_RESULTS.append({
                        "reference_dataset": reference_dataset,
                        "dataset": "all",
                        "score_name": score_name,
                        "k": k,
                        # "iw_elbo": iw_elbo, ??
                        # "iw_elbo_k": iw_elbo_k, ??
                        "AUROC": roc_auc,
                        "AUPRC": pr_auc,
                        "FPR80": fpr80,
                        "stat": score_name,
                        "run_id": run_id
                    })

                    if score_name in SCORES_TO_SHOW:
                        s += f"{score_name:10s}: {roc_auc:.3f} | "

                print(s)

        print("")

    results_df = pd.DataFrame(ALL_RESULTS)
    results_df.to_csv(os.path.join(wandb.run.dir, "results.csv"), index=False)
