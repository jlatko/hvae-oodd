GPU=1
#LIKELIHOOD="BernoulliLikelihoodConv2d"
#VAL_DATA="scripts/configs/val_datasets/binarized.json"

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
                               --run_name=resampled_hi \
                               --sampling_id=3r1vs6qa \
                               --sampling_key='FashionMNISTBinarized train' \
                               --sampling_mode=boost_high \
                               --sampling_a=600 \
                               --sampling_b=5 \
                               --train_datasets \
                               '{
                                   "FashionMNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets scripts/configs/val_datasets/binarized.json \
                               --likelihood BernoulliLikelihoodConv2d \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --run_name=resampled_pow_up \
                               --sampling_id=3r1vs6qa \
                               --sampling_key='FashionMNISTBinarized train' \
                               --sampling_mode=pow \
                               --sampling_a=100 \
                               --sampling_b=8 \
                               --train_datasets \
                               '{
                                   "FashionMNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets scripts/configs/val_datasets/binarized.json \
                               --likelihood BernoulliLikelihoodConv2d \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json

#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --run_name=resampled_lo \
#                               --sampling_id=3r1vs6qa \
#                               --sampling_key='FashionMNISTBinarized train' \
#                               --sampling_mode=boost_lo \
#                               --sampling_a=550 \
#                               --sampling_b=5 \
#                               --train_datasets \
#                               '{
#                                   "FashionMNISTBinarized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets scripts/configs/val_datasets/binarized.json \
#                               --likelihood BernoulliLikelihoodConv2d \
#                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json
