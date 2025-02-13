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
    minimum = np.max([np.min(count), 8])   # TODO: Magic number
    lines = count > 3 * minimum         # Lines are where the pixel count is more than 3 times the minimum
    filtered_lines = _filter_lines(lines)
    filtered_lines = _split_merged_lines(filtered_lines, count)
    return filtered_lines + separators[1]


def _filter_lines(mask):
    """
    Filter the mask to find the line boundaries, removes lines that are too short
    :param mask: the mask to filter
    :return: a list of tuples with the start and end of each line
    """
    line_boundaries = []
    in_line = False
    for i, value in enumerate(mask):        # Loop through the mask
        if value and not in_line:           # If the mask says there is a line and we are not already in a line
            in_line = True                  # Start a new line
            start = i                       # Set the start of the line
        elif not value and in_line:         # If the mask says there is no line and we are in a line
            in_line = False                 # End the line
            end = i                         # Set the end of the line
            if end - start > 15:            # If the line is longer than 15 pixels (in height) #TODO: Magic number
                line_boundaries.append((start, end))    # Add the line to the list of lines
    if in_line:                             # If we are still in a line at the end of the mask
        if len(mask) - start > 15:          # If the line is longer than 15 pixels (in height) #TODO: Magic number
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
            if pixel_count_per_line[i] == np.min(pixel_count_per_line[left:right]):  # If the pixel count is the minimum in the window
                if not minima or (i - minima[-1] > window_size):  # If the minimum is not too close to the previous one
                    minima.append(i)

        if len(minima) > 2: # If there are more than 2 minima, split the line
            for i in range(1, len(minima)): # Split the line at each minimum
                new_boundaries.append((minima[i - 1], minima[i]))
        else:   # If there are less than 2 minima, keep the line as is
            new_boundaries.append((start, end))
    return new_boundaries