# akd
python tools/train.py --cfg configs/cifar100/akd/res32x4_res8x4.yaml

# mlkd
python tools/train.py --cfg configs/cifar100/mlkd/res32x4_res8x4.yaml

# dkd
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml

# kd-logits-stand
python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 

# temp-scale
python tools/train.py --cfg configs/cifar100/akd/resnet32x4_resnet8x4.yaml --temp-scale
python tools/train.py --cfg configs/cifar100/akd/resnet32x4_resnet8x4.yaml --mlkd
