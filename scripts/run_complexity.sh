CUDA_VISIBLE_DEVICES='' python scripts/compute_entropies.py  \
--complexity=mean_local_entropy \
--complexity_param=3 \
--val_datasets scripts/configs/val_datasets/all.json


CUDA_VISIBLE_DEVICES='' python scripts/compute_entropies.py  \
--complexity=compression \
--complexity_param=0 \
--val_datasets  scripts/configs/val_datasets/all.json


CUDA_VISIBLE_DEVICES='' python scripts/compute_entropies.py  \
--complexity=compression \
--complexity_param=1 \
--val_datasets  scripts/configs/val_datasets/all.json