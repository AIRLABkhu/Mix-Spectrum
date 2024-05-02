# Mix-Spectrum for Generalization in Visual Reinforcement Learning
The official PytTorch implementation of "Mix-Spectrum for Generalization in Visual Reinforcement Learning"

## Install MuJoCo
Download the MuJoCo version 2.1 binaries for Linux or OSX. (https://www.roboti.us/)

Extract the downloaded mujoco210 directory into \~/.mujoco/mujoco210.

If you want to specify a nonstandard location for the package, use the env variable MUJOCO_PY_MUJOCO_PATH.  
pip3 install -U 'mujoco-py<2.2,>=2.1'


## Install DMControl
conda env create -f setup/conda.yml
conda activate dmcgb
sh setup/install_envs.sh


## Usage
## DMControl Benchmark

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


You can try other environments easily.



## Training
python src/train.py --domain_name walker --task_name walk --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 4 --gpu 0
python src/train.py --domain_name walker --task_name stand --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 4 --gpu 0
python src/train.py --domain_name cartpole --task_name swingup --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 8 --gpu 0
python src/train.py --domain_name ball_in_cup --task_name catch --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 4 --gpu 0
python src/train.py --domain_name finger --task_name spin --eval_mode color_easy --algorithm sac_aug --seed 1111 --augmentation mix_freq --action_repeat 2 --gpu 0
