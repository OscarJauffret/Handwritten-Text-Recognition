from skimage.filters import threshold_otsu
import numpy as np
from separate_page_sections import find_separators

def _binarize_image(image):
    """
    Binarize the image using Otsu's method
    :param image: the image to binarize
    :return: the binarized image
    """
    im = image.copy()
    threshold = threshold_otsu(im)
    return (im < threshold).astype(np.uint)

def _count_pixels_per_line(image):
    """
    Count the number of pixels per line in the image
    :param image: the image to count the pixels in
    :return: an array with the number of pixels per line
    """
    return np.sum(image == 1, axis=1)

def find_lines(image):
    """
    Find the handwritten lines boundaries in the image
    :param image: the image to find the lines in
    :return: a list of tuples with the start and end of each line
    """
    separators = find_separators(image)
    image = image[separators[1]:separators[2], :]
    count = _count_pixels_per_line(_binarize_image(image))
    minimum = np.max([np.min(count), 8])    # TODO: 8 is a magic number
    lines = count > 3 * minimum
    return _filter_lines(lines)

def _filter_lines(mask):
    """
    Filter the mask to find the line boundaries, removes lines that are too short
    :param mask: the mask to filter
    :return: a list of tuples with the start and end of each line
    """
    line_boundaries = []
    in_line = False
    for i, value in enumerate(mask):
        if value and not in_line:
            in_line = True
            start = i
        elif not value and in_line:
            in_line = False
            end = i
            if end - start > 15:
                line_boundaries.append((start, end))
    if in_line:
        if len(mask) - start > 15:
            line_boundaries.append((start, len(mask)))
    return line_boundaries
