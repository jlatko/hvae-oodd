GPU=1
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"


#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR100Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --special \
                               --seed 1 \
                               --run_name=2000_1_2_beta \
                               --epochs 2000 \
                               --max_beta 2 \
                               --train_datasets \
                               '{
                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json


