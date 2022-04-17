"""All configurations"""
import os

ROOT_FOLDER = "."


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


HAAR_CASCADE_EYE = "weights/haarcascade_eye.xml"
HAAR_CASCADE_FACE = "weights/haarcascade_frontalface_default.xml"

EYES_SAVE_PATH = "data/eyes/"
KMEANS_WEIGHTS = os.path.join(ROOT_FOLDER, "weights", "kmeans.pkl")
PCA_WEIGHTS = os.path.join(ROOT_FOLDER, "weights", "pca.pkl")
CARTOON_FEATURES_SAVE_PATH = os.path.join(
    "..", "..", "DeepL", "cartoongan", "data", "cartoon_features"
)
CARTOONS_CLUSTER_PATH = os.path.join("data", "cartoon_features.json")
