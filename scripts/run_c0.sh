GPU=0
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --epochs 1000 \
                               --train_datasets \
                               '{
                                   "FFHQ32Dequantized": {"split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --anneal \
#                               --swa \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#
#CUDA_VISIBLE_DEVICES=6 python scripts/dvae_run.py \
#                               --seed 1 \
#                               --anneal \
#                               --swa \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets scripts/configs/val_datasets/32_dequantized.json \
#                               --likelihood DiscretizedLogisticMixLikelihoodConv2d \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --run_name=4000_1 \
#                               --epochs 4000 \
#                               --seed 1 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#


