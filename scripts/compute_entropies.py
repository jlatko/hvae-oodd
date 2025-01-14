"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import io
import os
import logging

from collections import defaultdict

import wandb
from PIL import Image
from tqdm import tqdm

import rich
import numpy as np
import torch

import oodd.datasets
import oodd.utils

from skimage.filters.rank import entropy
from skimage.morphology import disk

from oodd.utils.wandb import download_or_find

from oodd.constants import WANDB_USER, WANDB_PROJECT, DATA_PATH

LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="FashionMNISTBinarized", help="model")
parser.add_argument("--complexity", type=str, default="mean_local_entropy", help="complexity metric")
parser.add_argument("--complexity_param", type=int, default=3, help="locality radius or compression mode")
parser.add_argument("--n_eval_examples", type=int, default=10000, help="cap on the number of examples to use")
parser.add_argument("--save_dir", type=str, default=f"{DATA_PATH}/oodd", help="directory to store scores in")
parser = oodd.datasets.DataModule.get_argparser(parents=[parser])

args = parser.parse_args()
rich.print(vars(args))


def get_save_path(name):
    name = name.replace(" ", "-")
    return f"{args.save_dir}/{name}"


def get_lengths(dataloaders):
    return [len(loader) for name, loader in dataloaders.items()]

def mean_local_entropy(x, radius=3):
    x = (x * 255).numpy().astype("uint8")
    entropies_per_channel = []
    for i in range(x.shape[0]):
        entropies_per_channel.append(
            np.mean(entropy(x[i], disk(radius)))
        )
    return np.mean(entropies_per_channel)

def get_size_bytesio(img, ext="JPEG", optimize=False):
    with io.BytesIO() as f:
        img.save(f, ext, optimize=optimize)
        s = f.getbuffer().nbytes
    return s

def compression(x, mode=0):
    x = (x * 255).numpy().astype("uint8")
    if x.shape[0] == 1:
        x = x[0]
    else:
        x = x.transpose(1, 2, 0)
    img = Image.fromarray(x)
    if mode == 0:  # JPEG optimized/not-optimized
        optimized = get_size_bytesio(img, ext="JPEG", optimize=True)
        unoptimized = get_size_bytesio(img, ext="JPEG", optimize=False)
        return optimized / unoptimized

    if mode == 1:  # JPEG optimized
        optimized = get_size_bytesio(img, ext="JPEG", optimize=True)
        return optimized

complexity_metrics = {
    "mean_local_entropy": mean_local_entropy,
    "compression": compression,
}

def init_wandb():
    tags = ["complexity"]
    wandb.init(project=WANDB_PROJECT, entity=WANDB_USER, dir=args.save_dir, tags=tags)
    args.save_dir = wandb.run.dir

    # wandb configuration
    run_name = "complexity_" + args.complexity + str(args.complexity_param) + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.config.update(args)
    wandb.run.save()

    # save data
    wandb.save("*.pt")

if __name__ == "__main__":
    init_wandb()

    # Data
    val_datasets = args.val_datasets
    datamodule = oodd.datasets.DataModule(
        batch_size=1,
        test_batch_size=1,
        data_workers=args.data_workers,
        train_datasets=[],
        val_datasets=val_datasets,
        test_datasets=[],
    )

    n_test_batches = get_lengths(datamodule.val_datasets)


    N_EQUAL_EXAMPLES_CAP = args.n_eval_examples
    LOGGER.info("%s = %s", "N_EQUAL_EXAMPLES_CAP", N_EQUAL_EXAMPLES_CAP)

    dataloaders = {(k + " " + val_datasets[k]["split"], v) for k, v in datamodule.val_loaders.items()}

    complexities = defaultdict(list)

    complexity_metric = complexity_metrics[args.complexity]

    for dataset, dataloader in dataloaders:
        # dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
        print(f"Evaluating {dataset}")

        n = 0
        for b, (x, _) in tqdm(enumerate(dataloader)):
            for xi in x:
                n += 1
                complexities[dataset].append(complexity_metric(xi, args.complexity_param))

                if n >= N_EQUAL_EXAMPLES_CAP:
                    LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP}")
                    break

            if n >= N_EQUAL_EXAMPLES_CAP:
                LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP}")
                break

        print(f"mean {args.complexity}({args.complexity_param}): ", np.mean(complexities[dataset]))
        wandb.log({
            f"{dataset} mean {args.complexity}({args.complexity_param}): ": np.mean(complexities[dataset]),
            f"{dataset} std {args.complexity}({args.complexity_param}): ": np.std(complexities[dataset]),
        })


    for dataset in sorted(complexities.keys()):
        print("===============", dataset, "===============")
        print(f"mean {args.complexity}({args.complexity_param}): ", np.mean(complexities[dataset]))
        print(f"std {args.complexity}({args.complexity_param}): ", np.std(complexities[dataset]))

    torch.save(complexities, get_save_path(f"complexity.pt"))