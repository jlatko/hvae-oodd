GPU=2
#LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2dMono"
LIKELIHOOD="GaussianLikelihoodConv2d"

VAL_DATA="scripts/configs/val_datasets/bnw_dequantized.json"

echo "Running experiments on GPU $GPU"


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "FashionMNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "MNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "KMNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json



CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "notMNISTDequentized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "Omniglot28x28Dequentized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "Omniglot28x28InvertedDequentized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=gaussian_05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "SmallNORB28x28Dequentized": {"split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json
