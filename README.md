# Mix-Spectrum for Generalization in Visual Reinforcement Learning
The official PytTorch implementation of "Mix-Spectrum for Generalization in Visual Reinforcement Learning". Note the main branch only contains the code for DMControl-GB. If you want the code for Procgen, you can find in Procgen branch.

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

## Framework
![overview](https://github.com/AIRLABkhu/Mix-Spectrum/assets/140928101/5129d59b-9a9d-41a0-86e6-aa96296c7b41)

## Experimental Results
### DMControl-GB
![Screenshot from 2024-05-02 10-54-15](https://github.com/AIRLABkhu/Mix-Spectrum/assets/140928101/b038d1f3-65a7-4860-9001-3ccf93b11e34)

### Procgen
![Screenshot from 2024-05-02 10-54-21](https://github.com/AIRLABkhu/Mix-Spectrum/assets/140928101/a5740779-9741-4d70-8138-86d3a68b4a42)

## Setup
### Requirements
- Ubuntu 20.04
- Python 3.7
- CUDA >=11.0

### Baselines
```bash
git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 
```
### Procgen
```bash
pip install procgen
```
### Python module requirements
```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensofrlow-gpu==2.5.1 
pip install gym==0.15.3
pip install higher==0.2 kornia==0.3.0
pip install tensorboard termcolor matplotlib imageio imageio-ffmpeg 
pip install scikit-image pandas pyyaml
```

## Training
```bash
python train.py --env_name $env --algo ppo --seed $seed --gpu_device $gpu
```


### Contact
For any questions, discussions, and proposals, please contact us at everyman123@khu.ac.kr

### Code Reference
- https://github.com/POSTECH-CVLab/style-agnostic-RL
