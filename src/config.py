import os
import string

class Config:
    file_name_regex = r"([a-z]\d{2}-\d{3}[a-z]?).png"

    class Paths:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        data_path = os.path.join(project_root, "data")
        sentences_path = os.path.join(data_path, "sentences.json")

        # This directory will contain .txt files containing the written texts.
        individual_sentences_path = os.path.join(data_path, "individual_sentences")

        # This directory should contain .xml files containing information about the written texts.
        images_meta_info_path = os.path.join(data_path, "images_meta_info")

        # This directory should contain the original images.
        original_images_path = os.path.join(data_path, "original_data")

        test = os.path.join(data_path, "test")
        test_images = os.path.join(test, "images")
        test_words = os.path.join(test, "words")
        test_labels = os.path.join(test, "labels")

        train = os.path.join(data_path, "train")
        train_images = os.path.join(train, "images")
        train_words = os.path.join(train, "words")
        train_labels = os.path.join(train, "labels")

        validate = os.path.join(data_path, "validate")
        validate_images = os.path.join(validate, "images")
        validate_words = os.path.join(validate, "words")
        validate_labels = os.path.join(validate, "labels")

        sentences_sizes_path = os.path.join(data_path, "sentences_sizes.xml")

        models_path = os.path.join(project_root, "models")

    class Colors:
        blue = '\033[94m'
        warning = '\033[93m'
        gray = '\033[90m'
        reset = '\033[0m'
        error = '\033[91m'

    class Data:
        line_width = 512
        line_height = 32
        word_width = 128
        word_height = 32

    class Model:
        output_width = 32
        alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,;'-()\"!&:?#+*/"
        epochs = -1     # Maximum number of epochs, put -1 for infinite (until validation is not done)
        learning_rate = 0.0001
        batch_size = 200
        num_workers = 0     # Use 0 because we are already preloading the dataset on GPU