import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sys
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import os
import cv2
import random
import time

places_dataloader = None
places_iter = None


def random_conv(x,args=None):
	"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
	n, c, h, w = x.shape
	for i in range(n):
		weights = torch.randn(3, 3, 3, 3).to(x.device)
		temp_x = x[i:i+1].reshape(-1, 3, h, w)
		temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
		out = torch.sigmoid(F.conv2d(temp_x, weights))
		total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
	return total_out.reshape(n, c, h, w)


def batch_from_obs(obs, args=None, batch_size=32):
    """Copy a single observation along the batch dimension"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)




def identity(x, args=None):
    return x


def random_shift(imgs, args=None, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
    return kornia.augmentation.RandomCrop((h, w))(imgs)


def random_window(imgs, args, min_cut=0, max_cut=84):
    """
    args:
    imgs: torch.Tensor shape (B, C, H, W)
    min / max cut: int, min / max size of cutout
    returns torch.Tensor
    """

    n, c, h, w = imgs.shape
    size = 64  

    w1 = torch.randint(0, w - size + 1, (n,))
    h1 = torch.randint(0, h - size + 1, (n,))

    cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype).to(torch.device("cuda:{}".format(args.gpu)))
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.clone()
        mask = torch.zeros_like(cut_img) 
        mask[:, h11:h11 + size, w11:w11 + size] = 1  
        cut_img = cut_img * mask 
        cutouts[i] = cut_img
    return cutouts


def random_crop(x, args=None, size=64, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
        'must either specify both w1 and h1 or neither of them'
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1
    cropped = F.interpolate(cropped, size=(84, 84), mode='bilinear', align_corners=False)

    return cropped


def view_as_windows_cuda(x, window_shape, args=None):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3)
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def random_flip(images, args=None, p=.2):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or gpu,
        p: prob of applying aug,
        returns torch.tensor
    """
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape

    images = images.to(device)

    flipped_images = images.flip([3])

    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1]  # // 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)

    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]

    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out


def random_rotation(images, args=None, p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: str, cpu or gpu,
        p: float, prob of applying aug,
        returns torch.tensor
    """
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape

    images = images.to(device)

    rot90_images = images.rot90(1, [2, 3])
    rot180_images = images.rot90(2, [2, 3])
    rot270_images = images.rot90(3, [2, 3])

    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)

    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i, m in enumerate(masks):
        m[torch.where(mask == i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:, :, None, None]
        masks[i] = m

    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, -1, h, w])
    return out


def random_cutout(imgs, args, min_cut=10, max_cut=30):
    """
    args:
    imgs: torch.Tensor shape (B, C, H, W)
    min / max cut: int, min / max size of cutout
    returns torch.Tensor
    """

    n, c, h, w = imgs.shape
    w1 = torch.randint(min_cut, max_cut, (n,))
    h1 = torch.randint(min_cut, max_cut, (n,))

    cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype).to(imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.clone()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        cutouts[i] = cut_img
    return cutouts


def random_cutout_color(imgs, args, min_cut=10, max_cut=30):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """

    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype).to(imgs.device)
    rand_box = torch.rand( size=(n, c), dtype=imgs.dtype).to(imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.clone()


        cut_img[:, h11:h11 + h11, w11:w11 + w11] = rand_box[i].view(-1, 1, 1).repeat(1, h11, w11)

        cutouts[i] = cut_img
    return cutouts


class Ppo():
    """
    Ppo
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr,
                 eps,
                 max_grad_norm):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.step = 0

    def make_mix_spectrum_samples(self,obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ):

        obs_batch_aug=obs_batch.clone()

        spectrum = torch.fft.fftn(obs_batch_aug, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

        # Get amplitude and Phase
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        B,C,H,W=amplitude.shape

        coeff=np.random.uniform(0.8,1.0,size=(B,))
        coeff=torch.FloatTensor(coeff).cuda()
        srm_out=torch.zeros_like(amplitude).cuda()

        x_C1=obs_batch.clone()
        x_C2=obs_batch.flip(0).clone()

        x_C2=random_conv(x_C2,None)


       	x_spectrum1=torch.fft.fftn(x_C1,dim=(-2,-1))
        x_spectrum1=torch.fft.fftshift(x_spectrum1,dim=(-2,-1))

        x_spectrum2=torch.fft.fftn(x_C2,dim=(-2,-1))
        x_spectrum2=torch.fft.fftshift(x_spectrum2,dim=(-2,-1))


        amplitude1=torch.abs(x_spectrum1)
        amplitude2=torch.abs(x_spectrum2)
        out_amplitude=amplitude1*(1-coeff).view(-1,1,1,1)+amplitude2*coeff.view(-1,1,1,1)
        out_spectrum=out_amplitude*torch.exp(1j*torch.angle(x_spectrum1))
        out_spectrum=torch.fft.ifftshift(out_spectrum,dim=(-2,-1))
        obs_aug=torch.fft.ifftn(out_spectrum,dim=(-2,-1)).float()


        obs_batch = obs_aug

        return obs_batch,recurrent_hidden_states_batch,actions_batch,value_preds_batch,return_batch,masks_batch,old_action_log_probs_batch,adv_targ
    def update(self, rollouts):
        self.step += 1

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                p=random.uniform(0,1)
                if p<0.5:
                    obs_batch, recurrent_hidden_states_batch, actions_batch, \
                        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ=self.make_mix_spectrum_samples(obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ)

                values, action_log_probs, dist_entropy, logits, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                
                
                # Update actor-critic using both PPO Loss
                self.optimizer.zero_grad()
                loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  
                    
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
