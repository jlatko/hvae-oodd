GPU=0
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2dMono"
VAL_DATA="scripts/configs/val_datasets/bnw_dequantized.json"

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
                               --run_name=test_mono \
                               --train_datasets \
                               '{
                                   "FashionMNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic_bnw.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic_bnw.json


#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --train_datasets \
#                               '{
#                                   "MNISTDequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic.json
#
#
#CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
#                               --seed 1 \
#                               --train_datasets \
#                               '{
#                                   "KMNISTDequantized": {"dynamic": true, "split": "train"}
#                               }' \
#                               --val_datasets $VAL_DATA \
#                               --likelihood $LIKELIHOOD \
#                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
#                               --config_stochastic scripts/configs/default_model/config_stochastic.json




