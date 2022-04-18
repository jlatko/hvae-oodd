import wandb
import os

from oodd.datasets.data_module import DATAMODULE_CONFIG_STR
from oodd.models.base_module import MODEL_CLASS_NAME_STR, MODEL_INIT_KWRGS_STR, MODEL_STATE_DICT_STR

REQUIRED_CHECKPOINT_FILES = [
    MODEL_CLASS_NAME_STR,
    MODEL_INIT_KWRGS_STR,
    MODEL_STATE_DICT_STR,
    DATAMODULE_CONFIG_STR
]
WANDB_PROJECT = ''
WANDB_USER = ''

def find_or_download_checkpoint(
        run_id=None,
        run=None,
        project=f"{WANDB_USER}/{WANDB_PROJECT}",
        force_redownload=False,
        target_dir="/scratch/s193223/oodd/wandb_downloads/"
):
    if run_id is not None:
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")
    else:
        run_id = run.id
    # check original wandb dirs
    found = [_find(filename, run) for filename in REQUIRED_CHECKPOINT_FILES]
    if all([pth is not None for pth in found]):
        return "/".join(found[0].split("/")[:-1])
    #otherwise download
    path = f"{target_dir}/{run_id}/"
    files = run.files()
    for filename in REQUIRED_CHECKPOINT_FILES:
        found = False
        for file in files:
            if file.name == filename:
                found = True
                _download(
                    file, path, force_redownload=force_redownload
                )
                break

        if not found:
            raise ValueError(f"{filename} not found in run {run_id}")
    return path

def download_or_find(
        run_id,
        filename,
        project=f"{WANDB_USER}/{WANDB_PROJECT}",
        force_redownload=False,
        target_dir="/scratch/s193223/oodd/wandb_downloads/"
):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    files = run.files()
    path = _find(filename, run, force_redownload=force_redownload)
    if path is not None:
        return path
    for file in files:
        if file.name == filename:
            return _download(
                file, f"{target_dir}/{run_id}/", force_redownload=force_redownload
            )

def _find(file_name, run, force_redownload=False):
    """ get from original run """
    if "save_dir" in run.config:
        full_path = os.path.join(run.config["save_dir"], file_name)
        if os.path.exists(full_path) and not force_redownload:
            return full_path
    return None

def _download(file, path, force_redownload=False):
    """ get from wandb or already downloaded """
    full_path = os.path.join(path, file.name)
    if os.path.exists(full_path) and not force_redownload:
        return full_path
    else:
        file.download(path, replace=True)
        return full_path