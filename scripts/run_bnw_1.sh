GPU=5
LIKELIHOOD="BernoulliLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/binarized.json"

echo "Running experiments on GPU $GPU"

#         'FashionMNISTBinarized': {'split': 'validation', 'dynamic': False},
#        'MNISTBinarized': {'split': 'validation', 'dynamic': False},
#        'notMNISTBinarized': {'split': 'validation'},
#        'Omniglot28x28Binarized': {'split': 'validation'},
#        'Omniglot28x28InvertedBinarized': {'split': 'validation'},
# not doing small norb as it's bad binarized
#        'SmallNORB28x28Binarized': {'split': 'validation'},
#        'KMNISTBinarized': {'split': 'validation', 'dynamic': False}


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "FashionMNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "MNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "KMNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "notMNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "Omniglot28x28Binarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "Omniglot28x28InvertedBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=05_nats \
                               --free_nats_end 0.5 \
                               --train_datasets \
                               '{
                                   "SmallNORB28x28Binarized": {"split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json
