# Policy Diffusion Network

## Installation

1. Clone the repository

2. Create a new Conda environment and activate it: 

```bash
conda create -n pdn python=3.10 -y
conda activate pdn
pip install -r requirements.txt
```

## training

### Step 1: train RL agents

We use stable-baselines3 and rlzoo to train RL policies:
```bash
python task_training.py task=data/CartPole-v1
python task_training.py task=data/Atari-Assault
python task_training.py task=data/Walker2d-v4
```

### Step 2: train diffusion models

The output is in `outputs/<task_name>` folder. The accuracy in logs means the return of the RL agent. 

train for single task:
```bash
python train.py system=multitask_ae_ddpm task=train/CartPole-v1 device.cuda_visible_devices=0
python train.py system=multitask_ae_ddpm task=train/Atari-Assault device.cuda_visible_devices=0
python train.py system=multitask_ae_ddpm task=train/Walker2d-v4 device.cuda_visible_devices=0

```
train for multitasks:
```bash
python train.py system=multitask_ae_ddpm task=train/mix system.ae_model.condition_num=3 system.model.condition_num=3 device.cuda_visible_devices=0
```
You need to modify condition_num to match the number of tasks. 

The example for multitasks is in configs/train/mix.yaml, you can modify it according to that format. 


