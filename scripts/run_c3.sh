GPU=3
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1000_1_beta_2_nats_05 \
                               --epochs 1000 \
                               --max_beta 2 \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --epochs 1000 \
                               --train_datasets \
                               '{
                                   "CIFAR10Dequantized": {"split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json

#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --special \
#                               --seed 1 \
#                               --run_name=2000_1_05_beta \
#                               --epochs 2000 \
#                               --max_beta 0.5 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json

#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --run_name=1000_1_big \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32_big.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32_big.json
#
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --run_name=4000_1_big \
#                               --epochs 4000 \
#                               --seed 1 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32_big.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32_big.json



