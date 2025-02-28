## Usage

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>) and [Multi-Level-Logit-Distillation](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation>).


### Installation

Environments:

- Python 3.8
- PyTorch 1.10.0
- torchvision 0.11.0
- CUDA 11.3

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

## Distilling CNNs

### CIFAR-100

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


Distill Student

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  # KD+Ours
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```


# Acknowledgement
Sincere gratitude to the contributors of mdistiller, CTKD, Multi-Level-Logit-Distillation and tiny-transformers for their distinguished efforts.
