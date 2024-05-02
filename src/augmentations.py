import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia
import utils
import os
import torchvision
import cv2
import random
import torch.fft
from torchvision.transforms import Resize
from torchvision.utils import save_image
import time

places_dataloader = None
places_iter = None


def _load_places(batch_size=256, image_size=84, num_workers=16, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in utils.load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x,args, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


def random_conv(x,args=None):
	"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
	n, c, h, w = x.shape
	for i in range(n):
		weights = torch.randn(3, 3, 3, 3).to(x.device)
		temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
		temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
		out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
		total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
	return total_out.reshape(n, c, h, w)


def batch_from_obs(obs, args=None,batch_size=32):
	"""Copy a single observation along the batch dimension"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, args=None,batch_size=32):
	"""Prepare batch for self-supervised policy adaptation at test-time"""
	batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
	batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
	batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

	return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def identity(x,args=None):
	return x


def random_shift(imgs,args=None, pad=4):
	"""Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
	_,_,h,w = imgs.shape
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
    size = 64  # 마스크 크기

    w1 = torch.randint(0, w - size + 1, (n,))
    h1 = torch.randint(0, h - size + 1, (n,))

    cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype).to(torch.device("cuda:{}".format(args.gpu)))
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.clone()
        mask = torch.zeros_like(cut_img)  # 마스크 생성, 모든 요소를 0으로 설정
        mask[:, h11:h11 + size, w11:w11 + size] = 1  # 영역을 1로 설정하여 마스크 생성
        cut_img = cut_img * mask  # 마스크와 요소별 곱셈을 통해 영역을 마스킹하여 0으로 만듦
        cutouts[i] = cut_img
    return cutouts
def random_crop(x,args=None, size=64, w1=None, h1=None, return_w1_h1=False):
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

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1
	cropped = F.interpolate(cropped, size=(84, 84), mode='bilinear', align_corners=False)

	return cropped


def view_as_windows_cuda(x, window_shape,args=None):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)


def random_flip(images,args=None, p=.2):
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


def random_rotation(images,args=None, p=.3):
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


def random_cutout_color(imgs,args, min_cut=10, max_cut=30):
	"""
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """

	n, c, h, w = imgs.shape
	w1 = np.random.randint(min_cut, max_cut, n)
	h1 = np.random.randint(min_cut, max_cut, n)

	cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype).to(imgs.device)
	rand_box = torch.randint(0, 255, size=(n, c), dtype=imgs.dtype).to(imgs.device)
	for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
		cut_img = img.clone()

		# 무작위 상자 추가
		cut_img[:, h11:h11 + h11, w11:w11 + w11] = rand_box[i].view(-1, 1, 1).repeat(1, h11, w11)

		cutouts[i] = cut_img
	return cutouts


def mix_freq(x,x2,args):
	p = random.uniform(0, 1)

	if p > 0.5:
		return x
	B,C,H,W=x.shape
	x2=random_conv(x2,args)

	coeff = np.random.uniform(0.8, 1.0, size=(B,))
	coeff=torch.FloatTensor(coeff).to(torch.device("cuda:{}".format(args.gpu)))
	srm_out=torch.zeros_like(x).to(torch.device("cuda:{}".format(args.gpu)))

	x_C1 = x
	x_C2 = x2

	x_spectrum1=torch.fft.fftn(x_C1,dim=(-2,-1))
	x_spectrum1=torch.fft.fftshift(x_spectrum1,dim=(-2,-1))

	x_spectrum2=torch.fft.fftn(x_C2,dim=(-2,-1))
	x_spectrum2=torch.fft.fftshift(x_spectrum2,dim=(-2,-1))

	amplitude1=torch.abs(x_spectrum1)
	amplitude2=torch.abs(x_spectrum2)
	out_amplitude=amplitude1*(1-coeff).view(-1,1,1,1)+amplitude2*coeff.view(-1,1,1,1)
	out_spectrum=out_amplitude*torch.exp(1j*torch.angle(x_spectrum1))
	out_spectrum=torch.fft.ifftshift(out_spectrum,dim=(-2,-1))
	srm_out=torch.fft.ifftn(out_spectrum,dim=(-2,-1)).float()

	return srm_out

def random_mask_freq_v1(x,args=None):
        p = random.uniform(0, 1)
        if p > 0.5:
             return x
        # need to adjust r1 r2 and delta for best performance
        r1=random.uniform(0,0.5)
        delta_r=random.uniform(0,0.035)
        r2=np.min((r1+delta_r,0.5))
        # print(r2)
        # generate Mask M
        B,C,H,W = x.shape
        center = (int(H/2), int(W/2))
        diagonal_lenth = max(H,W) # np.sqrt(H**2+W**2) is also ok, use a smaller r1
        r1_pix = diagonal_lenth * r1
        r2_pix = diagonal_lenth * r2
        Y_coord, X_coord = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((Y_coord - center[0])**2 + (X_coord - center[1])**2)
        M = dist_from_center <= r2_pix
        M = M * (dist_from_center >= r1_pix)
        M = ~M

        # mask Fourier spectrum
        M = torch.from_numpy(M).float().to(x.device)
        srm_out = torch.zeros_like(x)
        for i in range(C):
            x_c = x[:,i,:,:]
            x_spectrum = torch.fft.fftn(x_c, dim=(-2,-1))
            x_spectrum = torch.fft.fftshift(x_spectrum, dim=(-2,-1))
            out_spectrum = x_spectrum * M
            out_spectrum = torch.fft.ifftshift(out_spectrum, dim=(-2,-1))
            srm_out[:,i,:,:] = torch.fft.ifftn(out_spectrum, dim=(-2,-1)).float()
        return srm_out

