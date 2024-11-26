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
    minimum = np.max([np.min(count), 6])   # TODO: Magic number
    lines = count > 3 * minimum
    filtered_lines = _filter_lines(lines)
    filtered_lines = _split_merged_lines(filtered_lines, count)
    return filtered_lines


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


def _split_merged_lines(line_boundaries, pixel_count_per_line):
    """
    Split the lines that are too wide because they contain more than one line
    :param line_boundaries: the boundaries of the lines
    :param pixel_count_per_line: the number of pixels per line
    :return: the new boundaries of the lines
    """
    new_boundaries = []
    for start, end in line_boundaries:
        # Find the number of local minima in the window. If there are more than 2, split the line
        window_size = 40  # TODO: Magic number
        minima = []
        for i in range(start, end):
            left = max(start, i - window_size)
            right = min(end, i + window_size)
            if pixel_count_per_line[i] == np.min(
                    pixel_count_per_line[left:right]):  # If the pixel count is the minimum in the window
                if not minima or (i - minima[-1] > window_size):  #
                    minima.append(i)

        if len(minima) > 2:
            for i in range(1, len(minima)):
                new_boundaries.append((minima[i - 1], minima[i]))
        else:
            new_boundaries.append((start, end))
    return new_boundaries