"""All configurations"""
import os


BASE_DATA_PATH = "../../../Deepl/projet/cartoongan"  # "../../DeepL/cartoongan/data"

DATASETS_CSV_PATH = {
    "train": {
        "cartoons": os.path.join(BASE_DATA_PATH, "data/cartoons_train.csv"),
        "landscapes": os.path.join(BASE_DATA_PATH, "data/landscapes_train.csv"),
        "flickr": os.path.join(BASE_DATA_PATH, "data/pictures_train.csv"),
    },
    "validation": {
        "cartoons": os.path.join(BASE_DATA_PATH, "data/cartoons_validation.csv"),
        "landscapes": os.path.join(BASE_DATA_PATH, "data/landscapes_validation.csv"),
        "flickr": os.path.join(BASE_DATA_PATH, "data/pictures_validation.csv"),
    },
    "test": {
        "cartoons": os.path.join(BASE_DATA_PATH, "data/cartoons_test.csv"),
        "landscapes": os.path.join(BASE_DATA_PATH, "data/landscapes_test.csv"),
        "flickr": os.path.join(BASE_DATA_PATH, "data/pictures_test.csv"),
    },
}

REVERSED_CARTOON_HISTOGRAM_JSON = "data/reversed_cartoon_histogram.json"
