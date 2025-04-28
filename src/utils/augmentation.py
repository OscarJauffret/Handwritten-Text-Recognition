import torch
import kornia.augmentation as K

from kornia.morphology import dilation, erosion

class Augmenter:

    @staticmethod
    def dilate(tensor, kernel_size=3):
        kernel = torch.ones((kernel_size, kernel_size), device=tensor.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [B=1, C, H, W]
        return dilation(tensor, kernel).squeeze(0)

    @staticmethod
    def erode(tensor, kernel_size=3):
        kernel = torch.ones((kernel_size, kernel_size), device=tensor.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [B=1, C, H, W]
        return erosion(tensor, kernel).squeeze(0)

    @staticmethod
    def gamma_correction(image, gamma=0.5):
        # Assumes image is a torch.Tensor normalized between [0, 1]
        return image.pow(gamma)

    @staticmethod
    def pixel_dropout(image, dropout_prob=0.1, drop_value=1.0):
        # Randomly set some pixels to drop_value
        mask = (torch.rand_like(image) > dropout_prob).float()
        return image * mask + (1 - mask) * drop_value

    @staticmethod
    def add_gaussian_noise(image, std=0.3):
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)

    @staticmethod
    def apply_random_affine(image, max_translation=2, max_rotation=2):
        # Random affine using Kornia
        batch_image = image.unsqueeze(0)  # Add batch dimension
        aug = K.RandomAffine(degrees=max_rotation, translate=(max_translation/100, max_translation/100), p=1.0, padding_mode='border')
        out = aug(batch_image)
        return out.squeeze(0)