"""Predictor similar cartoon."""
import os
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from tqdm import tqdm

from src import config


class PredictorSimilarCartoon:
    """Predictor simimar cartoon."""

    def __init__(self) -> None:
        with open(config.KMEANS_WEIGHTS, "rb") as kmeans_file:
            self.kmeans = pickle.load(kmeans_file)

        with open(config.PCA_WEIGHTS, "rb") as pca_file:
            self.pca = pickle.load(pca_file)

        self.cartoons_paths, self.vectors = self.__get_cartoons_paths_and_vectors()
        self.cluster_groups = self.__get_cluster_groups()
        self.model = self.__load_model()

    @staticmethod
    def __load_model():
        """Load the model"""
        model = VGG16()
        # remove the output layer
        return Model(inputs=model.inputs, outputs=model.layers[-2].output)

    def __get_cluster_groups(self) -> Dict[int, List[str]]:
        """Get the cluster groups"""
        cluster_groups: Dict[int, List[str]] = {}
        for file_name, cluster in zip(self.cartoons_paths, self.kmeans.labels_):
            if cluster not in cluster_groups.keys():
                cluster_groups[cluster] = [file_name]
            else:
                cluster_groups[cluster].append(file_name)
        return cluster_groups

    @staticmethod
    def __get_cartoons_paths_and_vectors() -> Tuple[List[str], np.ndarray]:
        """Get the cartoons paths and vectors"""
        vector_space = []
        cartoon_names = []
        for folder in tqdm(os.listdir(config.CARTOON_FEATURES_SAVE_PATH)):
            folder_path = os.path.join(config.CARTOON_FEATURES_SAVE_PATH, folder)
            for npy_file in os.listdir(folder_path):
                npy_file_path = os.path.join(folder_path, npy_file)
                cartoon_names.append(npy_file_path)
                vector_space.append(np.load(npy_file_path))
        return cartoon_names, np.array(vector_space)

    @staticmethod
    def __compute_cosine_similarity(
        image_features: np.ndarray, cartoon_features: np.ndarray
    ) -> float:
        """Compute the cosine similarity"""
        cos_sim = np.dot(image_features, cartoon_features) / (
            np.linalg.norm(image_features) * np.linalg.norm(cartoon_features)
        )
        return float(cos_sim)

    def extract_features(self, image: np.ndarray):
        """Extracts features from an image"""

        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        reshaped_image = image.reshape(1, 224, 224, 3)  # pylint: disable=E1121
        image = preprocess_input(reshaped_image)
        return self.model.predict(image, use_multiprocessing=True)[0]

    def __get_best_cluster(self, image_features: np.ndarray) -> int:
        """Get the best cluster"""

        reduced_features = self.pca.transform(np.array([image_features]))

        return int(self.kmeans.predict(reduced_features)[0])

    def get_most_similar_image_path(self, image_path: str) -> Tuple[str, float]:
        """Get the most similar image"""
        image = np.array(load_img(image_path, target_size=(224, 224)))
        return self.get_n_most_similar_images(image, 1)[0]

    def get_most_similar_image_array(self, image: np.ndarray) -> Tuple[str, float]:
        """Get the most similar image"""
        return self.get_n_most_similar_images(image, 1)[0]

    def get_n_most_similar_images(
        self, image: str, number_images: int
    ) -> List[Tuple[str, float]]:
        """Get the n most similar images"""

        image_features = self.extract_features(image)

        best_cluster = self.__get_best_cluster(image_features)

        return sorted(
            [
                (
                    cartoon_path,
                    self.__compute_cosine_similarity(
                        image_features, np.load(cartoon_path)
                    ),
                )
                for cartoon_path in self.cluster_groups[best_cluster]
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:number_images]
