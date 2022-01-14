GPU=4
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"

# 1500_1_nats_1

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1500_1_nats_1 \
                               --epochs 1500 \
                               --free_nats_end 1 \
                               --train_datasets \
                               '{
                                   "SVHNDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json

# --- - 1500_1_nats_1_800

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1500_1_nats_1_800 \
                               --epochs 1500 \
                               --free_nats_end 1 \
                               --free_nats_epochs 800 \
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
                               --run_name=1500_1_nats_1_800 \
                               --epochs 1500 \
                               --free_nats_end 1 \
                               --free_nats_epochs 800 \
                               --train_datasets \
                               '{
                                   "FFHQ32Dequantized": { "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json

# ------ 1500_1_nats_2

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=1500_1_nats_2 \
                               --epochs 1500 \
                               --free_nats_end 2 \
                               --train_datasets \
                               '{
                                   "SVHNDequantized": {"dynamic": true, "split": "train"}
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
#                               --run_name=1500_1_beta_2_nats_1 \
#                               --epochs 1500 \
#                               --max_beta 2 \
#                               --free_nats_end 1 \
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
#                               --special \
#                               --seed 1 \
#                               --run_name=1500_1_beta_2_nats_1 \
#                               --epochs 1500 \
#                               --max_beta 2 \
#                               --free_nats_end 1 \
#                               --train_datasets \
#                               '{
#                                   "FFHQ32Dequantized": { "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --special \
#                               --seed 1 \
#                               --run_name=1500_1_beta_2_nats_1 \
#                               --epochs 1500 \
#                               --max_beta 2 \
#                               --free_nats_end 1 \
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
##VAL_DATA="scripts/configs/val_datasets/cifar_topics.json"
##
##
##CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
##                               --seed 1 \
##                               --train_datasets \
##                               '{
##                                   "CIFAR100DequantizedAnimals": {"dynamic": true, "split": "train"}
##                               }' \
##                               --val_datasets $VAL_DATA \
##                               --likelihood $LIKELIHOOD \
##                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
##                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
##
##
##CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
##                               --seed 1 \
##                               --train_datasets \
##                               '{
##                                   "CIFAR100DequantizedPlants": {"dynamic": true, "split": "train"}
##                               }' \
##                               --val_datasets $VAL_DATA \
##                               --likelihood $LIKELIHOOD \
##                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
##                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
#
#
#
#
