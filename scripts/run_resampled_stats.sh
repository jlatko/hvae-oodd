CUDA_VISIBLE_DEVICES=0 python scripts/ood-llr.py --run_ids=2fposcpq --run_name=fmnist_low \
                                  --use_train --val_datasets='{"FashionMNISTBinarized": {"split": "validation"}}'
CUDA_VISIBLE_DEVICES=0 python scripts/ood-llr.py --run_ids=28zlfebk --run_name=fmnist_hi \
                                  --use_train --val_datasets='{"FashionMNISTBinarized": {"split": "validation"}}'
CUDA_VISIBLE_DEVICES=0 python scripts/ood-llr.py --run_ids=29iu8ste --run_name=fmnist \
                                  --use_train --val_datasets='{"FashionMNISTBinarized": {"split": "validation"}}'

CUDA_VISIBLE_DEVICES=0 python scripts/ood-llr.py --run_ids=2atp7698 --run_name=cifar_low \
                                  --use_train --val_datasets='{"CIFAR10Dequantized": {"dynamic": false, "split": "validation"}}'
CUDA_VISIBLE_DEVICES=0 python scripts/ood-llr.py --run_ids=2876y4z7 --run_name=cifar_hi \
                                  --use_train --val_datasets='{"CIFAR10Dequantized": {"dynamic": false, "split": "validation"}}'
CUDA_VISIBLE_DEVICES=0 python scripts/ood-llr.py --run_ids=1vf6gfnd --run_name=cifar \
                                  --use_train --val_datasets='{"CIFAR10Dequantized": {"dynamic": false, "split": "validation"}}'