import numpy as np
from skimage.filters import sobel


def count_pixels_per_line(image):
    """
    Count the number of pixels per line in the image. This is useful to detect the header in the image.
    :param image: The image to count the pixels in.
    :return: An array with the number of pixels per line.
    """
    return np.sum(abs(sobel(image, axis=0)), axis=1)


def separate_header(image):
    """
    Finds the 3 horizontal lines in the image.
    They will allow us to separate the header from the rest of the image, and the rest of the image from the footer, reducing the amount of data to process.
    :param image: The array with the number of pixels per line.
    :return: The 3 largest peaks in the array.
    """
    px_per_line_array = count_pixels_per_line(image)

    # Find the 3 largest peaks
    # O(n) solution, since we know we are looking for 3 peaks
    # We know that the first two are in the first half of the image, and the last one is in the second half
    # We also know that the peaks are at least 100 pixels apart

    width = 100
    header_search_space = px_per_line_array[:len(px_per_line_array) // 2]
    top1 = np.argmax(header_search_space)
    y_from = max(0, top1 - width)
    y_to = min(len(header_search_space), top1 + width)
    header_search_space[y_from:y_to] = 0
    top2 = np.argmax(header_search_space)

    footer_search_space = px_per_line_array[len(px_per_line_array) // 2:]
    top3 = np.argmax(footer_search_space)

    # sort the 3 peaks
    top1, top2, top3 = sorted([top1, top2, top3])
    return top1, top2, top3