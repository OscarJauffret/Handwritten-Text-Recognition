import os
from os.path import split

from config import Config


def list_images(dir):
    """
    List all images in the directory
    :param dir: The directory to list images from
    :return: A list of image paths
    """
    files = []
    for root, _, file_names in os.walk(dir):
        if Config.Paths.sentences_path in root or Config.Paths.individual_sentences_path in root:
            continue
        for file_name in file_names:
            if file_name.endswith('.png'):
                files.append(os.path.join(root, file_name))
    return files


def split_data(nb_test, nb_validate, data_path):
    """
    Split the data into training, validation and testing sets. The values passed to this function are approximate because we do not want to split the data
    in the middle of a writer's images. Otherwise, the model would know the writer's style from the training set and would be able to recognize the writer
    :param nb_test: the approximate number of images to use for testing
    :param nb_validate: the approximate number of images to use for validation
    :param data_path: the path to the data directory
    :return: A tuple containing the training, validation and testing sets
    """
    images = list_images(data_path)
    test_images = extract_approx_n_images(data_path, images, nb_test)
    non_test_images = images[:-len(test_images)]
    validate_images = extract_approx_n_images(data_path, non_test_images, nb_validate)
    train_images = set(images) - test_images - validate_images


    return list(train_images), list(validate_images), list(test_images)

def extract_approx_n_images(data_path, images, nb_images):
    """
    Extract approximately n images from the data directory. The images are the last n images from the list of images passed to this function and all the images
    from the first writer in the list of images
    :param data_path: the path to the data directory
    :param images: the list of images to extract the images from
    :param nb_images: the approximate number of images to extract
    :return: A set of images
    """
    first_image = images[-nb_images]
    first_image_writer = first_image.split(os.sep)[-2]
    images_by_writer = []
    for image in os.listdir(os.path.join(data_path, first_image_writer)):
        if image.endswith('.png'):
            images_by_writer.append(os.path.join(data_path, first_image_writer, image))
    image_set = set(images[-nb_images:] + images_by_writer)
    return image_set


def move_image_to_folder(image_path, folder):
    """
    Move an image to a folder
    :param image_path: the path to the image
    :param folder: the folder to move the image to
    """
    image_name = split(image_path)[1]
    os.rename(image_path, os.path.join(folder, image_name))

def move_images_to_folder(images, folder):
    """
    Move a list of images to a folder
    :param images: the list of images to move
    :param folder: the folder to move the images to
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for image in images:
        move_image_to_folder(image, folder)


train_images, validate_images, test_images = split_data(50, 50, Config.Paths.original_images_path)

move_images_to_folder(train_images, Config.Paths.train_images)
move_images_to_folder(validate_images, Config.Paths.validate_images)
move_images_to_folder(test_images, Config.Paths.test_images)
