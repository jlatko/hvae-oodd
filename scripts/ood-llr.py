"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import os
import logging

from collections import defaultdict
from typing import *

from tqdm import tqdm

import rich
import numpy as np
import torch

import oodd
import oodd.datasets
import oodd.evaluators
import oodd.models
import oodd.losses
import oodd.utils
from oodd.utils import reduce_to_batch
import wandb

from oodd.utils.wandb import find_or_download_checkpoint

LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="wandb run id")
parser.add_argument("--iw_samples_elbo", type=int, default=1, help="importances samples for regular ELBO")
parser.add_argument("--iw_samples_Lk", type=int, default=1, help="importances samples for L>k bound")
parser.add_argument("--n_eval_examples", type=int, default=float("inf"), help="cap on the number of examples to use")
parser.add_argument("--n_latents_skip", type=int, default=1, help="the value of k in the paper")
parser.add_argument("--batch_size", type=int, default=500, help="batch size for evaluation")
parser.add_argument("--device", type=str, default="auto", help="device to evaluate on")
parser.add_argument("--save_dir", type=str, default= "/scratch/s193223/oodd", help="directory for saving results")

args = parser.parse_args()
rich.print(vars(args))

def load_run():
    api = wandb.Api()
    run = api.run(f"johnnysummer/hvae/{args.run_id}")
    checkpoint_path = find_or_download_checkpoint(run)
    return checkpoint_path, run


def setup_wandb(run):
    tags = [tag for tag in run.tags if tag != "train"]
    # add tags and initialize wandb run
    tags = [tags, "stats", args.run_id]

    wandb.init(project="hvae", entity="johnnysummer", dir=args.save_dir, tags=tags)
    args.save_dir = wandb.run.dir

    # wandb configuration
    run_name = "STATS_" + run.name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.config.update(args)
    wandb.config.update({
        "train_dataset": run.config['train_dataset'],
        "val_datasets": run.config['val_datasets'],
        "seed": run.config['seed'],
    })
    wandb.run.save()

    # save checkpoints
    wandb.save("*.pt")



def update_key(sample_stats, x, k):
    sample_stats[f"{k}_sum"].append(sum(x))
    for i, t in enumerate(x):
        sample_stats[f'{k}_{i}'].append(t)

def get_stage_stats(stage_data):
    return {
        "kl": (
            reduce_to_batch(stage_data.loss.kl_elementwise, batch_dim=0, reduction=torch.sum).detach()
            if stage_data.loss.kl_elementwise is not None
            else None
        ),
        "p_var": (
            reduce_to_batch(stage_data.p.variance, batch_dim=0, reduction=torch.mean).detach()
            if stage_data.p.variance is not None
            else None
        ),
        "q_var": (
            reduce_to_batch(stage_data.q.variance, batch_dim=0, reduction=torch.mean).detach()
            if stage_data.q.variance is not None
            else None
        ),
        "p_mean_sq": (
            reduce_to_batch(torch.pow(stage_data.p.mean, 2), batch_dim=0, reduction=torch.mean).detach()
            if stage_data.p.mean is not None
            else None
        ),
        "q_mean_sq": (
            reduce_to_batch(torch.pow(stage_data.q.mean, 2), batch_dim=0, reduction=torch.mean).detach()
            if stage_data.q.mean is not None
            else None
        ),
        "mean_diff_sq": (
            reduce_to_batch(torch.pow(stage_data.q.mean - stage_data.p.mean, 2), batch_dim=0, reduction=torch.mean).detach()
            if  (stage_data.q.mean is not None) and (stage_data.p.mean is not None)
            else None
        ),
        "var_diff_sq": (
            reduce_to_batch(torch.pow(stage_data.q.variance - stage_data.p.variance, 2), batch_dim=0, reduction=torch.mean).detach()
            if (stage_data.q.variance is not None) and (stage_data.p.variance is not None)
            else None
        ),
    }

def update_sample_stats(sample_stats, stage_datas):
    stages_stats = [get_stage_stats(stage_data) for stage_data in stage_datas]
    stats = {k: [dic[k] for dic in stages_stats if dic[k] is not None] for k in stages_stats[0]}
    for k, v in stats.items():
        update_key(sample_stats, v, k)

def stack_and_mean(stats):
    grouped_stats = {}
    # stats = {k: [dic[k] for dic in stats] for k in stats[0]}
    for k, v in stats.items():
        x = torch.stack(v, dim=0)
        grouped_stats[k] = torch.mean(x, dim=0)
    return grouped_stats


def get_stats_scores_sub(sample_stats, sample_stats_k):
    ks = set(sample_stats.keys()) & set(sample_stats_k.keys())
    return {k: sample_stats[k] - sample_stats_k[k] for k in ks}

def get_stats_scores_div(sample_stats, sample_stats_k):
    ks = set(sample_stats.keys()) & set(sample_stats_k.keys())
    return {k: sample_stats[k] / sample_stats_k[k] for k in ks}

def get_save_path(name):
    name = name.replace(" ", "-")
    return f"{args.save_dir}/{name}"


def get_decode_from_p(n_latents, k=0, semantic_k=True):
    """
    k semantic out
    0 True     [False, False, False]
    1 True     [True, False, False]
    2 True     [True, True, False]
    0 False    [True, True, True]
    1 False    [False, True, True]
    2 False    [False, False, True]
    """
    if semantic_k:
        return [True] * k + [False] * (n_latents - k)

    return [False] * (k + 1) + [True] * (n_latents - k - 1)


def get_lengths(dataloaders):
    return [len(loader) for name, loader in dataloaders.items()]


def print_stats(llr, l, lk):
    llr_mean, llr_var, llr_std = np.mean(llr), np.var(llr), np.std(llr)
    l_mean, l_var, l_std = np.mean(l), np.var(l), np.std(l)
    lk_mean, lk_var, lk_std = np.mean(lk), np.var(lk), np.std(lk)
    s = f"  {l_mean=:8.3f},   {l_var=:8.3f},   {l_std=:8.3f}\n"
    s += f"{llr_mean=:8.3f}, {llr_var=:8.3f}, {llr_std=:8.3f}\n"
    s += f" {lk_mean=:8.3f},  {lk_var=:8.3f},  {lk_std=:8.3f}"
    print(s)


def load_model_and_data(checkpoint_path):
    # Define checkpoints and load model
    checkpoint = oodd.models.Checkpoint(path=checkpoint_path)
    checkpoint.load()
    datamodule = checkpoint.datamodule
    model = checkpoint.model
    model.eval()
    rich.print(datamodule)
    # probably not helpful now
    datamodule.data_workers = 4
    datamodule.batch_size = args.batch_size
    datamodule.test_batch_size = args.batch_size
    LOGGER.info("%s", datamodule)
    dataloaders = {(k + " val", v) for k, v in datamodule.val_loaders.items()}
    dataloaders |= {(k + " test", v) for k, v in datamodule.test_loaders.items()}
    return model, datamodule, dataloader


if __name__ == "__main__":
    checkpoint_path, run = load_run()
    setup_wandb(run)
    model, datamodule, dataloaders = load_model_and_data(checkpoint_path)

    device = oodd.utils.get_device() if args.device == "auto" else torch.device(args.device)
    LOGGER.info("Device %s", device)

    FILE_NAME_SETTINGS_SPEC = f"k{args.n_latents_skip}-iw_elbo{args.iw_samples_elbo}-iw_lK{args.iw_samples_Lk}"

    # Add additional datasets to evaluation
    TRAIN_DATASET_KEY = list(datamodule.train_datasets.keys())[0]
    LOGGER.info("Train dataset %s", TRAIN_DATASET_KEY)

    MAIN_DATASET_NAME = list(datamodule.train_datasets.keys())[0].strip("Binarized").strip("Quantized").strip(
        "Dequantized")
    LOGGER.info("Main dataset %s", MAIN_DATASET_NAME)

    IN_DIST_DATASET = MAIN_DATASET_NAME + " test"
    TRAIN_DATASET = MAIN_DATASET_NAME + " train"
    LOGGER.info("Main in-distribution dataset %s", IN_DIST_DATASET)


    n_test_batches = get_lengths(datamodule.val_datasets) + get_lengths(datamodule.test_datasets)
    N_EQUAL_EXAMPLES_CAP = min(n_test_batches)
    assert N_EQUAL_EXAMPLES_CAP % args.batch_size == 0, "Batch size must divide smallest dataset size"

    N_EQUAL_EXAMPLES_CAP = min([args.n_eval_examples, N_EQUAL_EXAMPLES_CAP])
    LOGGER.info("%s = %s", "N_EQUAL_EXAMPLES_CAP", N_EQUAL_EXAMPLES_CAP)

    criterion = oodd.losses.ELBO()
    decode_from_p = get_decode_from_p(model.n_latents, k=args.n_latents_skip)


    scores = defaultdict(list)
    elbos = defaultdict(list)
    elbos_k = defaultdict(list)
    likelihoods = defaultdict(list)
    likelihoods_k = defaultdict(list)

    stats = defaultdict(lambda: defaultdict(list))
    stats_k = defaultdict(lambda: defaultdict(list))
    stats_scores_sub = defaultdict(lambda: defaultdict(list))
    stats_scores_div = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for dataset, dataloader in dataloaders:
            dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
            print(f"Evaluating {dataset}")

            n = 0
            for b, (x, _) in tqdm(enumerate(dataloader), total=N_EQUAL_EXAMPLES_CAP / args.batch_size):
                x = x.to(device)

                n += x.shape[0]
                sample_elbos, sample_elbos_k = [], []
                sample_likelihoods, sample_likelihoods_k = [], []
                sample_stats, sample_stats_k = defaultdict(list), defaultdict(list)

                # Regular ELBO
                for i in tqdm(range(args.iw_samples_elbo), leave=False):
                    likelihood_data, stage_datas = model(x, decode_from_p=False, use_mode=False)
                    kl_divergences = [
                        stage_data.loss.kl_elementwise
                        for stage_data in stage_datas
                        if stage_data.loss.kl_elementwise is not None
                    ]

                    update_sample_stats(sample_stats, stage_datas)

                    loss, elbo, likelihood, kl_divergences = criterion(
                        likelihood_data.likelihood,
                        kl_divergences,
                        samples=1,
                        free_nats=0,
                        beta=1,
                        sample_reduction=None,
                        batch_reduction=None,
                    )
                    sample_elbos.append(elbo.detach())
                    sample_likelihoods.append(likelihood.detach())

                # L>k bound
                for i in tqdm(range(args.iw_samples_Lk), leave=False):
                    likelihood_data_k, stage_datas_k = model(x, decode_from_p=decode_from_p, use_mode=decode_from_p)
                    kl_divergences_k = [
                        stage_data.loss.kl_elementwise
                        for stage_data in stage_datas
                        if stage_data.loss.kl_elementwise is not None
                    ]
                    loss_k, elbo_k, likelihood_k, kl_divergences_k = criterion(
                        likelihood_data_k.likelihood,
                        kl_divergences_k,
                        samples=1,
                        free_nats=0,
                        beta=1,
                        sample_reduction=None,
                        batch_reduction=None,
                    )

                    update_sample_stats(sample_stats_k, stage_datas_k)
                    sample_elbos_k.append(elbo_k.detach())
                    sample_likelihoods_k.append(likelihood_k.detach())

                sample_elbos = torch.stack(sample_elbos, axis=0)
                sample_elbos_k = torch.stack(sample_elbos_k, axis=0)

                sample_elbo = oodd.utils.log_sum_exp(sample_elbos, axis=0)
                sample_elbo_k = oodd.utils.log_sum_exp(sample_elbos_k, axis=0)

                sample_likelihoods = torch.stack(sample_likelihoods, axis=0)
                sample_likelihoods_k = torch.stack(sample_likelihoods_k, axis=0)
                sample_likelihoods = oodd.utils.log_sum_exp(sample_likelihoods, axis=0)
                sample_likelihoods_k = oodd.utils.log_sum_exp(sample_likelihoods_k, axis=0)

                sample_stats = stack_and_mean(sample_stats)
                sample_stats_k = stack_and_mean(sample_stats_k)

                sample_stats_scores_sub = get_stats_scores_sub(sample_stats, sample_stats_k)
                sample_stats_scores_div = get_stats_scores_div(sample_stats, sample_stats_k)
                score = sample_elbo - sample_elbo_k

                scores[dataset].extend(score.tolist())
                elbos[dataset].extend(sample_elbo.tolist())
                elbos_k[dataset].extend(sample_elbo_k.tolist())
                likelihoods[dataset].extend(sample_likelihoods.tolist())
                likelihoods_k[dataset].extend(sample_likelihoods_k.tolist())

                for k, v in sample_stats.items():
                    stats[dataset][k].extend(v.tolist())
                for k, v in sample_stats_k.items():
                    stats_k[dataset][k].extend(v.tolist())
                for k, v in sample_stats_scores_sub.items():
                    stats_scores_sub[dataset][k + '_sub'].extend(v.tolist())
                for k, v in sample_stats_scores_div.items():
                    stats_scores_div[dataset][k + '_div'].extend(v.tolist())

                if n > N_EQUAL_EXAMPLES_CAP:
                    LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP=}")
                    break

    # print likelihoods
    for dataset in sorted(scores.keys()):
        print("===============", dataset, "===============")
        print_stats(scores[dataset], elbos[dataset], elbos_k[dataset])

    # save scores
    torch.save(scores, get_save_path(f"values-scores-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save(elbos, get_save_path(f"values-elbos-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save(elbos_k, get_save_path(f"values-elbos_k-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save(likelihoods, get_save_path(f"values-likelihoods-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save(likelihoods_k, get_save_path(f"values-likelihoods_k-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save({f"{dataset}|{k}": v for dataset, d in stats.items() for k, v in d.items() }, get_save_path(f"values-stats-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save({f"{dataset}|{k}": v for dataset, d in stats_k.items() for k, v in d.items() }, get_save_path(f"values-stats_k-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save({f"{dataset}|{k}": v for dataset, d in stats_scores_sub.items() for k, v in d.items() }, get_save_path(f"values-stat_scores_sub-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
    torch.save({f"{dataset}|{k}": v for dataset, d in stats_scores_div.items() for k, v in d.items() }, get_save_path(f"values-stat_scores_div-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))