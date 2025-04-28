import numpy as np
from skimage.morphology import dilation, erosion, disk
from skimage import exposure
from skimage.filters import gaussian
from skimage.transform import AffineTransform, warp

class Augmenter:

    @staticmethod
    def dilate(image, size=1):
        return dilation(image, disk(size))

    @staticmethod
    def erode(image, size=1):
        return erosion(image, disk(size))

    @staticmethod
    def gamma_correction(image, gamma=0.5):
        return exposure.adjust_gamma(image, gamma)

    @staticmethod
    def pixel_dropout(image, dropout_prob=0.1, drop_value=255):
        mask = np.random.rand(*image.shape[:2]) < dropout_prob
        image = image.copy()
        if image.ndim == 2:
            image[mask] = drop_value
        else:
            for c in range(image.shape[2]):
                image[:, :, c][mask] = drop_value
        return image

    @staticmethod
    def add_gaussian_noise(image, sigma=0.3):
        return gaussian(image, sigma=sigma)

    @staticmethod
    def apply_random_affine(image, max_translation=2, max_rotation=2):
        translation = np.random.uniform(-max_translation, max_translation, size=2)
        rotation = np.deg2rad(np.random.uniform(-max_rotation, max_rotation))
        tform = AffineTransform(translation=translation, rotation=rotation)
        warped = warp(image, tform.inverse, mode='edge')
        return warped