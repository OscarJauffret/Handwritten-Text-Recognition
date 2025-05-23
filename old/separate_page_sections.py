import numpy as np
from skimage.filters import sobel


def detect_edges_per_line(image):
    """
    Detect the edges per line in the image using the Sobel filter.
    This is useful to detect the header in the image.
    :param image: The image to detect edges in.
    :return: An array with the number of edges per line.
    """
    return np.sum(abs(sobel(image, axis=0)), axis=1)

def find_separators(image):
    """
    Finds the 3 horizontal lines in the image.
    They will allow us to separate the header from the rest of the image, and the rest of the image from the footer, reducing the amount of data to process.
    :param image: The array with the number of pixels per line.
    :return: The 3 largest peaks in the array.
    """
    px_per_line_array = detect_edges_per_line(image)

    # Find the 3 largest peaks
    # We know that the first two are in the first half of the image, and the last one is in the second half
    # We also know that the peaks are at least 100 pixels apart, so once we find a peak, we can clear a window around it

    width = 100   # The width of the window to clear around the peak
    header_search_space = px_per_line_array[:len(px_per_line_array) // 2]   # Only search in the first half of the image for the header
    top1 = np.argmax(header_search_space)
    y_from = max(0, top1 - width)
    y_to = min(len(header_search_space), top1 + width)
    header_search_space[y_from:y_to] = 0    # Remove the peak we just found
    top2 = np.argmax(header_search_space)

    footer_search_space = px_per_line_array[len(px_per_line_array) // 2:]
    top3 = np.argmax(footer_search_space)
    # We need to add the length of the first half of the image to the index of the peak in the second half
    top3 += len(px_per_line_array) // 2

    # sort the 3 peaks
    top1, top2, top3 = sorted([top1, top2, top3])
    return top1, top2, top3