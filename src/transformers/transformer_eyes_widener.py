"""Transformer eyes widener"""
import os

import cv2
import numpy as np

from src import config
from src.base.image_array import ImageArray


class TransformerEyesWidener:
    """Transformer eyes widener"""

    def __init__(self, plot: bool = True) -> None:

        self.face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_FACE)
        self.eye_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_EYE)
        self.plot = plot

    def __call__(self, image_array: ImageArray) -> ImageArray:
        """
        Widens the eyes of the image.

        :param image_array: ImageArray
        :return: ImageArray
        """
        image = self.detect_eyes(image_array)
        self.show(image)

    def detect_eyes(  # pylint: disable=too-many-locals
        self, image_array: ImageArray
    ) -> ImageArray:
        """
        Detects the eyes of the image.

        :return: None
        """
        image = image_array.copy()

        faces = self.face_cascade.detectMultiScale(
            image, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) > 0:
            for (x, y, width, height) in faces:
                # print("face detected")

                # roi_face = gray_image[y : y + h, x : x + w]
                roi_face_color = image[y : y + height, x : x + width]

                eyes = self.eye_cascade.detectMultiScale(
                    roi_face_color, 1.01, 3, minSize=(10, 10)
                )
                for (eye_x, eye_y, eye_w, eye_h) in eyes:

                    roi = roi_face_color[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w]

                    # read eye
                    eye_image = cv2.imread(
                        os.path.join("data", "best_eyes", "black.png")
                    )
                    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
                    eye_image = cv2.resize(eye_image, (eye_w, eye_h))

                    # average value of initial roi
                    average_value = np.average(roi)
                    #  pixels with value [53, 66, 244] are replaced by average value
                    eye_image[
                        (eye_image[:, :, 0] == 0)
                        & (eye_image[:, :, 1] == 0)
                        & (eye_image[:, :, 2] == 0)
                    ] = average_value

                    # dilate region of interest
                    kernel = np.ones((3, 3), np.uint8)
                    roi = cv2.dilate(roi, kernel, iterations=1)

                    roi_face_color[
                        eye_y : eye_y + eye_h, eye_x : eye_x + eye_w
                    ] = cv2.addWeighted(eye_image, 0.5, roi, 0.5, 0)

                    image[y : y + height, x : x + width] = roi_face_color

        return image

    def show(self, image_array: ImageArray) -> ImageArray:
        """
        Shows the image.

        :return: None
        """
        if self.plot and image_array is not None:

            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", image_array)
            cv2.waitKey(0)
