"""All configurations"""
import os

ROOT_FOLDER = "."


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

KMEANS_WEIGHTS = os.path.join(ROOT_FOLDER, "weights", "kmeans.pkl")
PCA_WEIGHTS = os.path.join(ROOT_FOLDER, "weights", "pca.pkl")
CARTOON_FEATURES_SAVE_PATH = os.path.join(
    "..", "..", "DeepL", "cartoongan", "data", "cartoon_features"
)
CARTOONS_CLUSTER_PATH = os.path.join("data", "cartoon_features.json")
