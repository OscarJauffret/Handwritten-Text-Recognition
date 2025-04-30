import torch
import kornia.augmentation as K
import numpy as np

from kornia.morphology import dilation, erosion

from ..config import Config

class Augmenter:

    @staticmethod
    def augment(image_tensor):
        if np.random.random() < Config.Augmentation.Probs.dilate:
            image_tensor = Augmenter._dilate(image_tensor, Config.Augmentation.dilation_size)
        if np.random.random() < Config.Augmentation.Probs.erode:
            image_tensor = Augmenter._erode(image_tensor, Config.Augmentation.erosion_size)
        if np.random.random() < Config.Augmentation.Probs.gamma_correction:
            image_tensor = Augmenter._gamma_correction(image_tensor, Config.Augmentation.gamma)
        if np.random.random() < Config.Augmentation.Probs.pixel_dropout:
            image_tensor = Augmenter._pixel_dropout(image_tensor, Config.Augmentation.pixel_dropout_prob)
        if np.random.random() < Config.Augmentation.Probs.add_gaussian_noise:
            image_tensor = Augmenter._add_gaussian_noise(image_tensor, Config.Augmentation.gaussian_std)
        if np.random.random() < Config.Augmentation.Probs.apply_random_affine:
            image_tensor = Augmenter._apply_random_affine(image_tensor, Config.Augmentation.max_translation,
                                                     Config.Augmentation.max_rotation)
        return image_tensor

    @staticmethod
    def _dilate(tensor, kernel_size=3):
        kernel = torch.ones((kernel_size, kernel_size), device=tensor.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [B=1, C, H, W]
        return dilation(tensor, kernel).squeeze(0)

    @staticmethod
    def _erode(tensor, kernel_size=3):
        kernel = torch.ones((kernel_size, kernel_size), device=tensor.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [B=1, C, H, W]
        return erosion(tensor, kernel).squeeze(0)

    @staticmethod
    def _gamma_correction(image, gamma=0.5):
        # Assumes image is a torch.Tensor normalized between [0, 1]
        return image.pow(gamma)

    @staticmethod
    def _pixel_dropout(image, dropout_prob=0.1, drop_value=1.0):
        # Randomly set some pixels to drop_value
        mask = (torch.rand_like(image) > dropout_prob).float()
        return image * mask + (1 - mask) * drop_value

    @staticmethod
    def _add_gaussian_noise(image, std=0.3):
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)

    @staticmethod
    def _apply_random_affine(image, max_translation=2, max_rotation=2):
        # Random affine using Kornia
        batch_image = image.unsqueeze(0)  # Add batch dimension
        aug = K.RandomAffine(degrees=max_rotation, translate=(max_translation/100, max_translation/100), p=1.0, padding_mode='border')
        out = aug(batch_image)
        return out.squeeze(0)