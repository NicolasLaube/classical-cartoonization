"""All configurations"""


DATASETS_CSV_PATH = {
    "train": {
        "cartoons": "../../DeepL/cartoongan/data/cartoons_train.csv",
        "landscapes": "../../DeepL/cartoongan/data/landscapes_train.csv",
        "flickr": "../../DeepL/cartoongan/data/pictures_train.csv",
    },
    "validation": {
        "cartoons": "../../DeepL/cartoongan/data/cartoons_validation.csv",
        "landscapes": "../../DeepL/cartoongan/data/landscapes_validation.csv",
        "flickr": "../../DeepL/cartoongan/data/pictures_validation.csv",
    },
    "test": {
        "cartoons": "../../DeepL/cartoongan/data/cartoons_test.csv",
        "landscapes": "../../DeepL/cartoongan/data/landscapes_test.csv",
        "flickr": "../../DeepL/cartoongan/data/pictures_test.csv",
    },
}

REVERSED_HISTOGRAM_JSON = "data/histo_values.json"

BASE_DATA_PATH = "../../DeepL/cartoongan/"


HAAR_CASCADE_EYE = "weights/haarcascade_eye.xml"
HAAR_CASCADE_FACE = "weights/haarcascade_frontalface_default.xml"

EYES_SAVE_PATH = "data/eyes/"
