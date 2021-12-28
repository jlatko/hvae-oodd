GPU=3
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=1000_1_big \
                               --train_datasets \
                               '{
                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32_big.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32_big.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --run_name=4000_1_big \
                               --epochs 4000 \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32_big.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32_big.json



