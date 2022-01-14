GPU=2
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"

# -----


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1500_1_nats_1 \
                               --epochs 1500 \
                               --free_nats_end 1 \
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
                               --run_name=1500_1_nats_1 \
                               --epochs 1500 \
                               --free_nats_end 1 \
                               --train_datasets \
                               '{
                                   "FFHQ32Dequantized": { "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json

# -----

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1500_1_nats_1_800 \
                               --epochs 1500 \
                               --free_nats_end 1 \
                               --free_nats_epochs 800 \
                               --train_datasets \
                               '{
                                   "SVHNDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json


# -----

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1500_1_nats_2 \
                               --epochs 1500 \
                               --free_nats_end 2 \
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
                               --run_name=1500_1_nats_2 \
                               --epochs 1500 \
                               --free_nats_end 2 \
                               --train_datasets \
                               '{
                                   "FFHQ32Dequantized": { "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json




#
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --special \
#                               --seed 1 \
#                               --run_name=1000_1_beta_2_nats_05 \
#                               --epochs 1000 \
#                               --max_beta 2 \
#                               --free_nats_end 0.5 \
#                               --train_datasets \
#                               '{
#                                   "SVHNDequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --special \
#                               --seed 1 \
#                               --epochs 1000 \
#                               --train_datasets \
#                               '{
#                                   "SVHNDequantized": {"split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --special \
#                               --seed 1 \
#                               --run_name=1000_1_nats_05 \
#                               --epochs 1000 \
#                               --free_nats_end 0.5 \
#                               --train_datasets \
#                               '{
#                                   "SVHNDequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --train_datasets \
#                               '{
#                                   "SVHNDequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --run_name=1000_2 \
#                               --seed 2 \
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
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --run_name=1000_3 \
#                               --seed 3 \
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
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --run_name=1000_4 \
#                               --seed 4 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json



