import argparse
import datetime
import logging
import os

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR

from oodd.utils.wandb import download_or_find
from oodd.constants import WANDB_USER, WANDB_PROJECT, DATA_PATH

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import matplotlib.pyplot as plt
import numpy as np
import rich
import sklearn.metrics
import torch
import torch.utils.data
import wandb
from tqdm import tqdm

import oodd
import oodd.models
import oodd.datasets
import oodd.variational
import oodd.losses
from oodd.datasets.data_module import parse_dataset_argument, get_sample_weights

from oodd.utils import str2bool, get_device, log_sum_exp, set_seed, plot_gallery
from oodd.evaluators import Evaluator


LOGGER = logging.getLogger(name=__file__)


parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument("--model", default="VAE", help="model type (vae | lvae | biva)")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
parser.add_argument("--samples", type=int, default=1, help="samples from approximate posterior")
parser.add_argument("--importance_weighted", type=str2bool, default=False, const=True, nargs="?", help="use iw bound")
parser.add_argument("--warmup_epochs", type=int, default=200, help="epochs to warm up the KL term.")
parser.add_argument("--max_beta", type=float, default=1, help="maximum beta term")
parser.add_argument("--free_nats_epochs", type=int, default=400, help="epochs to warm up the KL term.")
parser.add_argument("--free_nats", type=float, default=2, help="nats considered free in the KL term")
parser.add_argument("--free_nats_end", type=float, default=0.0, help="final free nats")
parser.add_argument("--n_eval_samples", type=int, default=32, help="samples from prior for quality inspection")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed")
parser.add_argument("--test_every", type=int, default=20, help="epochs between evaluations")
parser.add_argument("--save_dir", type=str, default= f"{DATA_PATH}/oodd", help="directory for saving models")
parser.add_argument("--tqdm", action= "store_true", help="whether to display progressbar")
parser.add_argument("--run_name", type=str, default=None, help="name this wandb run")
parser.add_argument("--test_verbosity", type=int, default=1, help="how much test values to log")

parser.add_argument("--sampling_id", type=str, default=None, help="")
parser.add_argument("--sampling_key", type=str, default=None, help="")
parser.add_argument("--sampling_mode", type=str, default="pow", help="")
parser.add_argument("--sampling_a", type=float, default=100, help="")
parser.add_argument("--sampling_b", type=float, default=5, help="")

parser.add_argument("--anneal", action= "store_true", help="use CosineAnnealingLR")
parser.add_argument("--swa", action= "store_true", help="use SWA") # not working
parser.add_argument("--swa_start", type=int, default=600, help="SWA start epoch") # not working

# tags
TAG_FLAGS = ["special", "test_run", "final"]
for tag in TAG_FLAGS:
    parser.add_argument(f"--{tag}", action= "store_true", help="add tag") # not working


parser = oodd.datasets.DataModule.get_argparser(parents=[parser])

args, unknown_args = parser.parse_known_args()

args.start_time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
args.sample_reduction = log_sum_exp if args.importance_weighted else torch.mean

set_seed(args.seed)
device = get_device()

def setup_wandb(train_dataset_name):
    # add tags and initialize wandb run
    tags = [train_dataset_name, f"seed_{args.seed}", "train"]

    dargs = vars(args)

    if args.sampling_id is not None:
        tags.append(f"sampl_{args.sampling_mode}")

    for key in TAG_FLAGS:
        if dargs[key]:
            tags.append(key)

    wandb.init(project=WANDB_PROJECT, entity=WANDB_USER, dir=args.save_dir, tags=tags)
    args.save_dir = wandb.run.dir

    # wandb configuration
    run_name = ""
    if args.run_name is not None:
        run_name = "_" + args.run_name
    if wandb.run.name is not None:
        run_name = train_dataset_name + run_name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.config.update(args)
    wandb.config.update({"train_dataset": train_dataset_name})
    wandb.run.save()

    # save checkpoints
    wandb.save("*.pt")
    wandb.save("*.npy")


def train(epoch):
    model.train()
    evaluator = Evaluator(primary_metric="log p(x)", logger=LOGGER)

    beta = next(deterministic_warmup)
    free_nats = next(free_nats_cooldown)

    iterator = tqdm(enumerate(datamodule.train_loader), smoothing=0.9, total=len(datamodule.train_loader), leave=False, disable=(not args.tqdm))
    s = time.time()
    inds = []
    for _, (x, idx) in iterator:
        x = x.to(device)
        inds.extend(list(idx.detach().cpu().numpy()))
        likelihood_data, stage_datas = model(x, n_posterior_samples=args.samples)
        kl_divergences = [
            stage_data.loss.kl_elementwise for stage_data in stage_datas if stage_data.loss.kl_elementwise is not None
        ]
        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.samples,
            free_nats=free_nats,
            beta=beta,
            sample_reduction=args.sample_reduction,
            batch_reduction=None,
        )

        l = loss.mean()
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        evaluator.update("Train", "elbo", {"log p(x)": elbo})
        evaluator.update("Train", "likelihoods", {"loss": -loss, "log p(x)": elbo, "log p(x|z)": likelihood})
        klds = {
            f"KL z{i+1}": kl
            for i, kl in enumerate([sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None])
        }
        klds["KL(q(z|x), p(z))"] = kl_divergences
        evaluator.update("Train", "divergences", klds)

    # log time
    time_passed = time.time() - s
    print(f"Took: {time_passed:.2f} seconds")
    wandb.log({"training time": time_passed, "epoch": epoch}, step=epoch * len(datamodule.train_loader))

    evaluator.update(
        "Train", "hyperparameters", {"free_nats": [free_nats], "beta": [beta], "learning_rate": [optimizer.param_groups[0]['lr']]}
    )
    evaluator.report(epoch * len(datamodule.train_loader))
    evaluator.log(epoch)

    np.save(os.path.join(wandb.run.dir, f"inds_{epoch}.npy"), np.array(inds))

    if args.swa and epoch >args.swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    elif scheduler is not None:
        print("STEP: scheduler LR: ", scheduler.get_last_lr(), "optimizer.param_groups[0]['lr']", optimizer.param_groups[0]['lr'])
        scheduler.step()
        wandb.log({"scheduler_lr": scheduler.get_last_lr()}, step=epoch * len(datamodule.train_loader))


@torch.no_grad()
def test(model, epoch, dataloader, evaluator, dataset_name="test", max_test_examples=float("inf"), is_main=False):
    LOGGER.info(f"Testing: {dataset_name}")
    model.eval()

    # visualize reconstructions
    x, _ = next(iter(dataloader))
    x = x.to(device)
    n = min(x.size(0), 8)
    likelihood_data, stage_datas = model(x, n_posterior_samples=args.samples)
    p_x_mean = likelihood_data.mean[: args.batch_size].view(args.batch_size, *in_shape)  # Reshape the zeroth "sample"
    p_x_samples = likelihood_data.samples[: args.batch_size].view(
        args.batch_size, *in_shape
    )  # Reshape the zeroth "sample"

    decode_from_p = [True] * (model.n_latents - 1) + [False]
    likelihood_data, stage_datas = model(x, n_posterior_samples=args.samples, decode_from_p=decode_from_p, use_mode=decode_from_p)
    p_x_mean_top = likelihood_data.mean[: args.batch_size].view(args.batch_size, *in_shape)  # Reshape the zeroth "sample"
    p_x_samples_top = likelihood_data.samples[: args.batch_size].view(
        args.batch_size, *in_shape
    )  # Reshape the zeroth "sample"

    comparison = torch.cat([x[:n], p_x_mean[:n], p_x_samples[:n], p_x_mean_top[:n], p_x_samples_top[:n]])
    comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
    fig, ax, img = plot_gallery(comparison.cpu().numpy(), ncols=n)
    fig.savefig(os.path.join(args.save_dir, f"reconstructions_{dataset_name}_{epoch:03}"))
    plt.close()
    images = wandb.Image(img, caption=f"reconstructions_{dataset_name}_{epoch:03}")
    wandb.log({f"reconstructions_{dataset_name}": images}, step=epoch * len(datamodule.train_loader))

    # Evaluations
    decode_from_p_combinations = [[True] * n_p + [False] * (model.n_latents - n_p) for n_p in range(model.n_latents)]
    for decode_from_p in tqdm(decode_from_p_combinations, leave=False, disable=(not args.tqdm)):
        n_skipped_latents = sum(decode_from_p)

        if max_test_examples != float("inf"):
            iterator = tqdm(
                zip(range(max_test_examples // dataloader.batch_size), dataloader),
                smoothing=0.9,
                total=max_test_examples // dataloader.batch_size,
                leave=False,
                disable=(not args.tqdm)
            )
        else:
            iterator = tqdm(enumerate(dataloader), smoothing=0.9, total=len(dataloader), leave=False, disable=(not args.tqdm))

        for _, (x, _) in iterator:
            x = x.to(device)

            likelihood_data, stage_datas = model(
                x, n_posterior_samples=args.samples, decode_from_p=decode_from_p, use_mode=decode_from_p
            )
            kl_divergences = [
                stage_data.loss.kl_elementwise
                for stage_data in stage_datas
                if stage_data.loss.kl_elementwise is not None
            ]
            loss, elbo, likelihood, kl_divergences = criterion(
                likelihood_data.likelihood,
                kl_divergences,
                samples=args.samples,
                free_nats=0,
                beta=1,
                sample_reduction=args.sample_reduction,
                batch_reduction=None,
            )

            if n_skipped_latents == 0:  # Regular ELBO
                evaluator.update(dataset_name, "elbo", {"log p(x)": elbo})

                if is_main or args.test_verbosity > 1:
                    evaluator.update(
                        dataset_name, "likelihoods", {"loss": -loss, "log p(x)": elbo, "log p(x|z)": likelihood}
                    )
                    klds = {
                        f"KL z{i+1}": kl
                        for i, kl in enumerate(
                            [sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None]
                        )
                    }
                    klds["KL(q(z|x), p(z))"] = kl_divergences
                    evaluator.update(dataset_name, "divergences", klds)

            evaluator.update(dataset_name, f"skip-elbo", {f"{n_skipped_latents} log p(x)": elbo})
            evaluator.update(dataset_name, f"skip-elbo-{dataset_name}", {f"{n_skipped_latents} log p(x)": elbo})

            if is_main or args.test_verbosity > 1:
                evaluator.update(
                    dataset_name,
                    f"skip-likelihoods-{dataset_name}",
                    {
                        f"{n_skipped_latents} loss": -loss,
                        f"{n_skipped_latents} log p(x)": elbo,
                        f"{n_skipped_latents} log p(x|z)": likelihood,
                    },
                )
                klds = {
                    f"{n_skipped_latents} KL z{i+1}": kl
                    for i, kl in enumerate(
                        [sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None]
                    )
                }
                klds[f"{n_skipped_latents} KL(q(z|x), p(z))"] = kl_divergences
                evaluator.update(dataset_name, f"skip-divergences-{dataset_name}", klds)


def collapse_multiclass_to_binary(y_true, zero_label=None):
    # Force the class index in zero_label to be zero and the others to collapse to 1
    zero_label_indices = y_true == zero_label
    y_true[zero_label_indices] = 0
    y_true[~zero_label_indices] = 1
    return y_true


def compute_roc_auc(y_true=None, y_score=None, zero_label=None):
    """Compares class zero_label to all other classes in y_true"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score, average="macro")
    return roc_auc, fpr, tpr, thresholds


def compute_pr_auc(y_true=None, y_score=None, zero_label=None):
    """Compares class zero_label to all other classes in y_true"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score, average="macro")
    return pr_auc, precision, recall, thresholds


def compute_roc_pr_metrics(y_true, y_score, classes, reference_class):
    """Compute the ROC and PR metrics from a primary dataset class to a number of other dataset classes"""
    roc_results = {}
    pr_results = {}
    for class_label in sorted(set(y_true) - set([reference_class])):
        idx = np.logical_or(y_true == reference_class, y_true == class_label)  # Compare primary to the other dataset

        roc_auc, fpr, tpr, thresholds = compute_roc_auc(
            y_true=y_true[idx], y_score=y_score[idx], zero_label=reference_class
        )

        pr_auc, precision, recall, thresholds = compute_pr_auc(
            y_true=y_true[idx], y_score=y_score[idx], zero_label=reference_class
        )

        idx_where_tpr_is_eighty = np.where((tpr - 0.8 >= 0))[0][0]
        fpr80 = fpr[idx_where_tpr_is_eighty]

        ood_target = [source for source, label in classes.items() if label == class_label][0]
        roc_results[ood_target] = dict(roc_auc=roc_auc, fpr=fpr, tpr=tpr, fpr80=fpr80, thresholds=thresholds)
        pr_results[ood_target] = dict(pr_auc=pr_auc, precision=precision, recall=recall, thresholds=thresholds)

    return roc_results, pr_results


def subsample_labels_and_scores(y_true, y_score, n_examples):
    """Subsample y_true and y_score to have n_examples while maintaining their relative ordering"""
    assert len(y_true) == len(y_score) >= n_examples, f"Got {len(y_true)=}, {len(y_score)=}, {n_examples=}"
    indices = [np.random.choice(np.where(y_true == i)[0], n_examples, replace=False) for i in set(y_true)]
    y_true = np.concatenate([y_true[idx] for idx in indices])
    y_score = np.concatenate([y_score[idx] for idx in indices])
    return y_true, y_score

def get_sampling_weights(args):
    if args.sampling_id is None:
        return None
    else:
        assert args.sampling_key is not None

        path = download_or_find(args.sampling_id, "complexity.pt")
        complexities = torch.load(path)
        print(complexities.keys())
        complexities = np.array(complexities[args.sampling_key])
        return get_sample_weights(complexities, args.sampling_mode, args.sampling_a, args.sampling_b)

if __name__ == "__main__":
    # Data
    sampling_weights = get_sampling_weights(args)
    datamodule = oodd.datasets.DataModule(
        batch_size=args.batch_size,
        test_batch_size=250,
        data_workers=args.data_workers,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        test_datasets=args.test_datasets,
        wrap_datasets=True,
        sample_weigths=sampling_weights
    )
    train_dataset_name = list(datamodule.train_datasets.keys())[0]
    setup_wandb(train_dataset_name)
    # os.makedirs(args.save_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(args.save_dir, "dvae.log"))
    fh.setLevel(logging.INFO)
    LOGGER.addHandler(fh)

    in_shape = datamodule.train_dataset.datasets[0].size[0]
    datamodule.save(args.save_dir)

    # Model
    model = getattr(oodd.models.dvae, args.model)
    model_argparser = model.get_argparser()
    model_args, unknown_model_args = model_argparser.parse_known_args()
    model_args.input_shape = in_shape
    # update wandb
    wandb.config.update(model_args)
    wandb.run.save()

    model = model(**vars(model_args)).to(device)

    p_z_samples = model.prior.sample(torch.Size([args.n_eval_samples])).to(device)
    sample_latents = [None] * (model.n_latents - 1) + [p_z_samples]

    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.anneal:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    if args.swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    criterion = oodd.losses.ELBO()

    deterministic_warmup = oodd.variational.DeterministicWarmup(n=args.warmup_epochs, t_max=args.max_beta)
    free_nats_cooldown = oodd.variational.FreeNatsCooldown(
        constant_epochs=args.free_nats_epochs // 2,
        cooldown_epochs=args.free_nats_epochs // 2,
        start_val=args.free_nats,
        end_val=args.free_nats_end,
    )

    # Logging
    LOGGER.info("Experiment config:")
    LOGGER.info(args)
    rich.print(vars(args))
    LOGGER.info("%s", deterministic_warmup)
    LOGGER.info("%s", free_nats_cooldown)
    LOGGER.info("DataModule:\n%s", datamodule)
    LOGGER.info("Model:\n%s", model)

    # Run
    test_elbos = [-np.inf]
    test_evaluator = Evaluator(primary_source=datamodule.primary_val_name, primary_metric="log p(x)", logger=LOGGER)

    LOGGER.info("Running training...")
    for epoch in range(1, args.epochs + 1):
        train(epoch)

        if args.swa and epoch > args.swa_start:
            model_to_use = swa_model
        else:
            model_to_use = model

        if epoch % args.test_every == 0:
            # Sample
            with torch.no_grad():
                likelihood_data, stage_datas = model_to_use.sample_from_prior(
                    n_prior_samples=args.n_eval_samples, forced_latent=sample_latents
                )
                p_x_samples = likelihood_data.samples.view(args.n_eval_samples, *in_shape)
                p_x_mean = likelihood_data.mean.view(args.n_eval_samples, *in_shape)
                comparison = torch.cat([p_x_samples, p_x_mean])
                comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
                fig, ax, img = plot_gallery(comparison.cpu().numpy(), ncols=args.n_eval_samples // 4)
                fig.savefig(os.path.join(args.save_dir, f"samples_{epoch:03}"))
                plt.close()
                images = wandb.Image(img, caption=f"samples_{epoch:03}")
                wandb.log({"samples": images}, step=epoch * len(datamodule.train_loader))

            # Test
            for name, dataloader in datamodule.val_loaders.items():
                test(model_to_use, epoch, dataloader=dataloader, evaluator=test_evaluator, dataset_name=name, max_test_examples=10000,
                     is_main=(name == datamodule.primary_val_name))

            # Save
            test_elbo = test_evaluator.get_primary_metric().mean().cpu().numpy()
            if np.max(test_elbos) < test_elbo:
                test_evaluator.save(args.save_dir)
                model.save(args.save_dir)
                if args.swa and epoch >args.swa_start:
                    swa_model.save(args.save_dir)
                LOGGER.info("Saved model!")
            test_elbos.append(test_elbo)

            # Compute LLR
            for source in test_evaluator.sources:
                for k in range(1, model.n_latents):
                    log_p_a = test_evaluator.metrics[source][f"skip-elbo"][f"0 log p(x)"]
                    log_p_b = test_evaluator.metrics[source][f"skip-elbo"][f"{k} log p(x)"]
                    llr = log_p_a - log_p_b
                    test_evaluator.update(source, series="LLR", metrics={f"LLR>{k}": llr})

            # Compute AUROC score for L>k and LLR>k metrics
            reference_dataset = datamodule.primary_val_name
            max_examples = min(
                [len(d) for d in datamodule.val_datasets.values()]
            )  # Maximum number of examples to use for equal sized sets

            # L >k
            for n_skipped_latents in range(model.n_latents):
                y_true, y_score, classes = test_evaluator.get_classes_and_scores_per_source(
                    f"skip-elbo", f"{n_skipped_latents} log p(x)"
                )
                y_true, y_score = subsample_labels_and_scores(y_true, y_score, max_examples)
                roc, pr = compute_roc_pr_metrics(
                    y_true, -y_score, classes, classes[reference_dataset]
                )  # Negation since higher score means more OOD
                for ood_target, value_dict in roc.items():
                    # test_evaluator.update(
                    #     source=reference_dataset,
                    #     series=f"ROC AUC L>k",
                    #     metrics={f"ROC AUC L>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    # )
                    test_evaluator.update(
                        source=reference_dataset,
                        series=f"ROC AUC L>{n_skipped_latents}",
                        metrics={f"ROC AUC L>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    )

            # LLR >0 >k
            for n_skipped_latents in range(1, model.n_latents):
                y_true, y_score, classes = test_evaluator.get_classes_and_scores_per_source(
                    f"LLR", f"LLR>{n_skipped_latents}"
                )
                y_true, y_score = subsample_labels_and_scores(y_true, y_score, max_examples)
                roc, pr = compute_roc_pr_metrics(y_true, y_score, classes, classes[reference_dataset])
                for ood_target, value_dict in roc.items():
                    # test_evaluator.update(
                    #     source=reference_dataset,
                    #     series=f"ROC AUC LLR>k",
                    #     metrics={f"ROC AUC LLR>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    # )
                    test_evaluator.update(
                        source=reference_dataset,
                        series=f"ROC AUC LLR>{n_skipped_latents}",
                        metrics={f"ROC AUC LLR>{n_skipped_latents} {ood_target}": [value_dict["roc_auc"]]},
                    )

            # Report
            test_evaluator.report(epoch * len(datamodule.train_loader))
            test_evaluator.log(epoch)
            test_evaluator.reset()
