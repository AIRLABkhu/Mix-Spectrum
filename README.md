# Mix-Spectrum for Generalization in Visual Reinforcement Learning
The official PytTorch implementation of "Mix-Spectrum for Generalization in Visual Reinforcement Learning"

## Abstract
> Visual Reinforcement Learning (RL) trains agents on policies using raw images showing potential for real-world applications. 
However, the limited diversity in the training environment often results in overfitting with agents underperforming in unseen environments.
To address this issue, image augmentation is utilized in visual RL, but the effectiveness is limited due to the potential to alter the state information of the image.
Therefore, we introduce \textit{Mix-Spectrum}, a straightforward yet highly effective frequency-based augmentation method that maintains the state consistency of data and enhances the agent's focus on semantic information.
The proposed method mixes the original and reference image randomly sampled from the dataset in the frequency domain.
Our method transforms the images to the Fourier frequency domain using the Fast Fourier Transform (FFT) and mixes only the amplitudes while preserving the phase of the original image.
Furthermore, to introduce the diversity of amplitude, our method initially applies Random Convolution to the reference image as a perturbation in the frequency domain.
These allow the augmentation of both preserving the semantic information and increasing the diversity of amplitude for robust generalization in visual RL tasks.
Furthermore, the proposed method stands out for adaptability integrating with any visual RL algorithm, whether off-policy or on-policy.
Through extensive experiments on the DMControl Generalization Benchmark (DMControl-GB) and Procgen, our method demonstrates superior performance compared to existing frequency-based and image augmentation methods in zero-shot generalization.

## Figure
![overview](https://github.com/AIRLABkhu/Mix-Spectrum/assets/140928101/5129d59b-9a9d-41a0-86e6-aa96296c7b41)


## Setup
### Install MuJoCo
Download the MuJoCo version 2.1 binaries for Linux or OSX. (https://www.roboti.us/)

Extract the downloaded mujoco210 directory into \~/.mujoco/mujoco210.

If you want to specify a nonstandard location for the package, use the env variable MUJOCO_PY_MUJOCO_PATH.  
pip3 install -U 'mujoco-py<2.2,>=2.1'


### Install DMControl

``` bash
conda env create -f setup/conda.yml
conda activate dmcgb
sh setup/install_envs.sh
```

### Usage
``` bash
from env.wrappers import make_env  
env = make_env(  
        domain_name=args.domain_name,  
        task_name=args.task_name,  
        seed=args.seed,  
        episode_length=args.episode_length,  
        action_repeat=args.action_repeat,  
        image_size=args.image_size,  
        mode='train'  
)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)  
```

You can try other environments easily.



### Training
``` bash
python src/train.py --domain_name walker --task_name walk --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 4 --gpu 0
python src/train.py --domain_name walker --task_name stand --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 4 --gpu 0
python src/train.py --domain_name cartpole --task_name swingup --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 8 --gpu 0
python src/train.py --domain_name ball_in_cup --task_name catch --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 4 --gpu 0
python src/train.py --domain_name finger --task_name spin --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 2 --gpu 0
```
