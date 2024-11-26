from tkinter import image_names

from skimage.filters import threshold_otsu, rank as skr
import matplotlib.pyplot as plt
from skimage import io, color
import os
from config import Config
from skimage.morphology import disk
import numpy


def median_filter(image, disk_size):
    m = skr.median(image, disk(disk_size))
    return m

def otsus_threshold(image):
    if image.ndim == 3:
        image = color.rgb2gray(image)

    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value

    count_white = numpy.sum(binary_image > 0)
    count_black = numpy.sum(binary_image == 0)

    # if there are more black pixels than whites, then it's the background that is black so we invert the image's color
    if count_black > count_white:
        binary_image = 255 - binary_image

    return binary_image

image_name = "a01-077u"
image_path = os.path.join(Config.Paths.data_path, "000", f"{image_name}.png")
image = io.imread(image_path)

median_filtered_image = median_filter(image, 3)
io.imsave(os.path.join(Config.Paths.project_root, "output", f"{image_name}_median.png"), median_filtered_image)

binary_image = otsus_threshold(median_filtered_image)
io.imsave(os.path.join(Config.Paths.project_root, "output", f"{image_name}_binary.png"), (binary_image * 255).astype('uint8'))

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# ax = axes.ravel()
# ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].set_title("Original Image")
# ax[1].imshow(binary_image, cmap=plt.cm.gray)
# ax[1].set_title("Binary Image (Otsu's threshold)")
#
# for a in ax:
#     a.axis('off')
#
# plt.tight_layout()
# plt.show()