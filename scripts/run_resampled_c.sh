GPU=5
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/32_dequantized.json"

echo "Running experiments on GPU $GPU"

#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --run_name=test__nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --special \
#                               --epochs 10 \
#                               --free_nats_end 0.5 \
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
#                               --run_name=test_resampled_hi_nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --special \
#                               --epochs 10 \
#                               --sampling_mode=boost_high \
#                               --sampling_a=600 \
#                               --sampling_b=5 \
#                               --free_nats_end 0.5 \
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
#                               --run_name=test_resampled_pow_hi_nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --special \
#                               --epochs 10 \
#                               --sampling_mode=pow \
#                               --sampling_a=100 \
#                               --sampling_b=9 \
#                               --free_nats_end 0.5 \
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
#                               --run_name=test_resampled_pow_lo_nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --special \
#                               --epochs 10 \
#                               --sampling_mode=pow \
#                               --sampling_a=100 \
#                               --sampling_b=-9 \
#                               --free_nats_end 0.5 \
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
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --run_name=test_resampled_low_nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --special \
#                               --epochs 10 \
#                               --sampling_mode=boost_low \
#                               --sampling_a=550 \
#                               --sampling_b=5 \
#                               --free_nats_end 0.5 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json


#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --run_name=resampled_very_low_nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --sampling_mode=boost_low \
#                               --sampling_a=575 \
#                               --sampling_b=30 \
#                               --free_nats_end 0.5 \
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
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --run_name=resampled_very_high_nats_05 \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='CIFAR10Dequantized train' \
#                               --sampling_mode=boost_high \
#                               --sampling_a=650 \
#                               --sampling_b=30 \
#                               --free_nats_end 0.5 \
#                               --train_datasets \
#                               '{
#                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=resampled_pow_very_hi_nats_05 \
                               --sampling_id=3r1vs6qa \
                               --sampling_key='CIFAR10Dequantized train' \
                               --sampling_mode=pow \
                               --sampling_a=100 \
                               --sampling_b=11 \
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
                               --seed 1 \
                               --run_name=resampled_pow_very_lo_nats_05 \
                               --sampling_id=3r1vs6qa \
                               --sampling_key='CIFAR10Dequantized train' \
                               --sampling_mode=pow \
                               --sampling_a=100 \
                               --sampling_b=-11 \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "CIFAR10Dequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_32.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_32.json
