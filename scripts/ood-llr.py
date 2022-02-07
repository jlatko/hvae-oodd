"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import json
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
from oodd.datasets import DataModule
from oodd.utils import reduce_to_batch
import wandb

from oodd.utils.argparsing import json_file_or_json_unique_keys
from oodd.utils.wandb import find_or_download_checkpoint

LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--run_ids", type=str, help="wandb run id or a list")
parser.add_argument("--run_name", type=str, default=None, help="name this wandb run")
parser.add_argument("--iw_samples_elbo", type=int, default=1, help="importances samples for regular ELBO")
parser.add_argument("--iw_samples_Lk", type=int, default=1, help="importances samples for L>k bound")
parser.add_argument("--n_eval_examples", type=int, default=10000, help="cap on the number of examples to use")
parser.add_argument("--batch_size", type=int, default=500, help="batch size for evaluation")
parser.add_argument("--device", type=str, default="auto", help="device to evaluate on")
parser.add_argument("--use_test", action="store_true")
parser.add_argument("--use_train", action="store_true")
parser.add_argument("--val_datasets", type=json_file_or_json_unique_keys, default=None)
parser.add_argument("--save_dir", type=str, default= "/scratch/s193223/oodd", help="directory for saving results")

args = parser.parse_args()
rich.print(vars(args))

def get_all_configs():
    configs = {}
    for cfg_file in os.listdir("scripts/configs/val_datasets/"):
        if cfg_file != "all.json":
            with open(os.path.join("scripts/configs/val_datasets/", cfg_file), 'r') as fh:
                configs[cfg_file] = json.load(fh)
    return configs

def load_run(run_id):
    api = wandb.Api()
    run = api.run(f"johnnysummer/hvae/{run_id}")
    checkpoint_path = find_or_download_checkpoint(run=run)
    return checkpoint_path, run


def setup_wandb(run=None):
    # add tags and initialize wandb run
    tags = ["stats"]
    if run is not None:
        tags += [tag for tag in run.tags if tag != "train"]

    wandb.init(project="hvae", entity="johnnysummer", dir=args.save_dir, tags=tags)
    args.save_dir = wandb.run.dir
    wandb.config.update(args)

    # wandb configuration

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = "multiple"

    run_name = "STATS_" + run_name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
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

def get_dataset_config(main_dataset):
    if args.val_datasets is not None:
        return args.val_datasets
    all_val_configs = get_all_configs()
    for c, conf in all_val_configs.items():
        if main_dataset in conf:
            return c, conf


def load_model_and_data(run_id):
    checkpoint_path, run = load_run(run_id=run_id)

    # Define checkpoints and load model
    print("LOADING checkpoint from: ", checkpoint_path)
    checkpoint = oodd.models.Checkpoint(path=checkpoint_path)
    checkpoint.load()
    main_dataset = list(run.config['train_datasets'].keys())[0]
    print("Main (train) dataset: ", main_dataset)
    c, datasets = get_dataset_config(main_dataset)
    print("Will use dataset config from: ", c)
    print(datasets)


    if args.use_test:
        datasets = run.config['val_datasets'].copy()
        for k in datasets.keys():
            datasets[k]['split'] = "test"

        datamodule = DataModule(
            train_datasets=run.config['train_datasets'],
            val_datasets=[],
            test_datasets=datasets,
        )
    else:
        datamodule = DataModule(
            train_datasets=run.config['train_datasets'],
            val_datasets=datasets,
            test_datasets=[],
        )
    model = checkpoint.model
    model.eval()
    rich.print(datamodule)
    # probably not helpful now
    datamodule.data_workers = 4
    datamodule.batch_size = args.batch_size
    datamodule.test_batch_size = args.batch_size
    LOGGER.info("%s", datamodule)

    if args.use_test:
        dataloaders = {(k + " test", v) for k, v in datamodule.test_loaders.items()}
    else:
        dataloaders = {(k + " val", v) for k, v in datamodule.val_loaders.items()}

    if args.use_train:
        dataloaders |= {(k + " train", v) for k, v in datamodule.train_loaders.items()}

    return model, datamodule, dataloaders, main_dataset


def get_elbo_and_stats(x, model, criterion, decode_from_p, use_mode):
    sample_elbos = []
    sample_likelihoods = []
    sample_kls = []
    sample_stats = defaultdict(list)

    for i in tqdm(range(args.iw_samples_elbo), leave=False):
        likelihood_data, stage_datas = model(x, decode_from_p=decode_from_p, use_mode=use_mode)
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
        sample_kls.append(kl_divergences.detach())

    sample_elbos = torch.stack(sample_elbos, axis=0)
    sample_elbo = oodd.utils.log_sum_exp(sample_elbos, axis=0)
    sample_likelihoods = torch.stack(sample_likelihoods, axis=0)
    sample_likelihoods = oodd.utils.log_sum_exp(sample_likelihoods, axis=0)
    sample_kls = torch.stack(sample_kls, axis=0)
    sample_kls = oodd.utils.log_sum_exp(sample_kls, axis=0)
    sample_stats = stack_and_mean(sample_stats)

    return sample_elbo, sample_likelihoods, sample_kls, sample_stats

def get_all_stats_for_sample(x, model, criterion, TRAIN_DATASET_KEY, dataset, run_id):
    (
        sample_elbo,
        sample_likelihoods,
        sample_kls,
        sample_stats
    ) = get_elbo_and_stats(x, model, criterion, False, False)

    # dataset, k, score name
    all_scores[TRAIN_DATASET_KEY][dataset][0][run_id]['ELBO'].extend(sample_elbo.tolist())
    all_scores[TRAIN_DATASET_KEY][dataset][0][run_id]['LIKELIHOOD'].extend(sample_likelihoods.tolist())
    all_scores[TRAIN_DATASET_KEY][dataset][0][run_id]['KL'].extend(sample_kls.tolist())
    for stat, v in sample_stats.items():
        all_scores[TRAIN_DATASET_KEY][dataset][0][run_id][stat].extend(v.tolist())

    for k in range(1, model.n_latents):
        decode_from_p = get_decode_from_p(model.n_latents, k=k)
        (
            sample_elbo_k,
            sample_likelihoods_k,
            sample_kls_k,
            sample_stats_k
        ) = get_elbo_and_stats(x, model, criterion, decode_from_p=decode_from_p, use_mode=decode_from_p)

        # dataset, k, score name
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['ELBO'].extend(sample_elbo_k.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['LIKELIHOOD'].extend(sample_likelihoods_k.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['KL'].extend(sample_kls_k.tolist())
        for stat, v in sample_stats_k.items():
            all_scores[TRAIN_DATASET_KEY][dataset][k][run_id][stat].extend(v.tolist())

        # Get ratio scores
        LLR = sample_elbo - sample_elbo_k
        LIKELIHOOD_RATIO = sample_likelihoods - sample_likelihoods_k
        KL_RATIO = sample_kls - sample_kls_k

        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['LLR'].extend(LLR.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['LIKELIHOOD_RATIO'].extend(LIKELIHOOD_RATIO.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['KL_RATIO'].extend(KL_RATIO.tolist())

        # also for other stats?
        sample_stats_scores_sub = get_stats_scores_sub(sample_stats, sample_stats_k)
        sample_stats_scores_div = get_stats_scores_div(sample_stats, sample_stats_k)
        for stat, v in sample_stats_scores_sub.items():
            all_scores[TRAIN_DATASET_KEY][dataset][k][run_id][stat + '_sub'].extend(v.tolist())
        for stat, v in sample_stats_scores_div.items():
            all_scores[TRAIN_DATASET_KEY][dataset][k][run_id][stat + '_div'].extend(v.tolist())


def get_simple_stats_for_sample(x, model, criterion, TRAIN_DATASET_KEY, dataset, run_id, run_mode):
    # TODO: modes
    # Run without variances:
    # - Don’t sample but use variance for KL A
    # - Don’t sample and don’t use variance for KL B
    # - Sample but use fixed variance C
    suffix = " " + run_mode
    if run_mode in ["A"]:
        use_mode = True
    else:
        use_mode = False

    (
        sample_elbo,
        sample_likelihoods,
        sample_kls,
        sample_stats
    ) = get_elbo_and_stats(x, model, criterion, False, use_mode)

    # dataset, k, score name
    all_scores[TRAIN_DATASET_KEY][dataset][0][run_id]['ELBO' + suffix].extend(sample_elbo.tolist())
    all_scores[TRAIN_DATASET_KEY][dataset][0][run_id]['LIKELIHOOD' + suffix].extend(sample_likelihoods.tolist())
    all_scores[TRAIN_DATASET_KEY][dataset][0][run_id]['KL' + suffix].extend(sample_kls.tolist())

    for k in range(1, model.n_latents):
        decode_from_p = get_decode_from_p(model.n_latents, k=k)
        if run_mode == "A":
            use_mode = True
        elif run_mode == "B":
            use_mode = False
        else:
            use_mode = decode_from_p

        (
            sample_elbo_k,
            sample_likelihoods_k,
            sample_kls_k,
            sample_stats_k
        ) = get_elbo_and_stats(x, model, criterion, decode_from_p=decode_from_p, use_mode=use_mode)

        # dataset, k, score name
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['ELBO' + suffix].extend(sample_elbo_k.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['LIKELIHOOD' + suffix].extend(sample_likelihoods_k.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['KL'+ suffix].extend(sample_kls_k.tolist())

        # Get ratio scores
        LLR = sample_elbo - sample_elbo_k
        LIKELIHOOD_RATIO = sample_likelihoods - sample_likelihoods_k
        KL_RATIO = sample_kls - sample_kls_k

        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['LLR' + suffix].extend(LLR.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['LIKELIHOOD_RATIO' + suffix].extend(LIKELIHOOD_RATIO.tolist())
        all_scores[TRAIN_DATASET_KEY][dataset][k][run_id]['KL_RATIO' + suffix].extend(KL_RATIO.tolist())


def main():
    # FILE_NAME_SETTINGS_SPEC = f"iw_elbo{args.iw_samples_elbo}-iw_lK{args.iw_samples_Lk}"

    # train dataset, val datasets, k, run id (in case multiple seeds), stat name

    run_ids = args.run_ids.split(",")
    setup_wandb()

    for run_id in run_ids:
        LOGGER.info("RUN ID %s", run_id)

        model, datamodule, dataloaders, main_dataset = load_model_and_data(run_id)

        device = oodd.utils.get_device() if args.device == "auto" else torch.device(args.device)
        LOGGER.info("Device %s", device)

        # Add additional datasets to evaluation
        TRAIN_DATASET_KEY = main_dataset

        LOGGER.info("Train dataset %s", TRAIN_DATASET_KEY)

        n_test_batches = get_lengths(datamodule.val_datasets) + get_lengths(datamodule.test_datasets)
        N_EQUAL_EXAMPLES_CAP = args.n_eval_examples

        criterion = oodd.losses.ELBO()

        with torch.no_grad():
            for dataset, dataloader in dataloaders:
                # dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
                print(f"Evaluating {dataset}")

                n = 0
                for b, (x, _) in tqdm(enumerate(dataloader), total=N_EQUAL_EXAMPLES_CAP / args.batch_size):
                    x = x.to(device)

                    n += x.shape[0]

                    get_all_stats_for_sample(x, model, criterion, TRAIN_DATASET_KEY, dataset, run_id)
                    get_simple_stats_for_sample(x, model, criterion, TRAIN_DATASET_KEY, dataset, run_id, "A")
                    get_simple_stats_for_sample(x, model, criterion, TRAIN_DATASET_KEY, dataset, run_id, "B")

                    if n > N_EQUAL_EXAMPLES_CAP:
                        LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP=}")
                        break

    torch.save({
        in_dataset: {
            dataset: {
                k: {
                    run_id: {
                        stat: value for stat, value in d4.items()
                    } for run_id, d4 in d3.items()
                } for k, d3 in d2.items()
            } for dataset, d2 in d1.items()
        } for in_dataset, d1 in all_scores.items()
    }, get_save_path("all-scores.pt"))


if __name__ == "__main__":
    # GLOBALS
    all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))


    main()
