"""Transformer eyes widener"""
import os

import cv2

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
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 5, 1, 1)
        self.detect_eyes(image_array, gray_image)
        # self.widen_eyes(image_array)
        # self.show(image_array)

    def detect_eyes(
        self, image_array: ImageArray, gray_image: ImageArray
    ) -> ImageArray:
        """
        Detects the eyes of the image.

        :return: None
        """
        image = image_array.copy()

        faces = self.face_cascade.detectMultiScale(
            image, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)
        )
        if len(faces) > 0:
            for (x, y, width, height) in faces:
                # print("face detected")

                cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

                # roi_face = gray_image[y : y + h, x : x + w]
                roi_face_color = image_array[y : y + height, x : x + width]

                eyes = self.eye_cascade.detectMultiScale(
                    roi_face_color, 1.01, 3, minSize=(10, 10)
                )
                for (eye_x, eye_y, eye_w, eye_h) in eyes:
                    # print("eye detected")
                    # cv2.rectangle(
                    #     roi_face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    # )
                    # save eye
                    cv2.imwrite(
                        os.path.join(
                            config.EYES_SAVE_PATH,
                            f"{len(os.listdir(config.EYES_SAVE_PATH)) + 1}.jpg",
                        ),
                        roi_face_color[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w],
                    )
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image with bounding boxes", image)
        # cv2.waitKey(0)

        return gray_image

    # def widen_eyes(self, image_array: ImageArray) -> ImageArray:
    #     """
    #     Widens the eyes of the image.

    #     :return: None
    #     """
    #     pass

    def show(self, image_array: ImageArray) -> ImageArray:
        """
        Shows the image.

        :return: None
        """
        if self.plot:
            cv2.imshow("Image", image_array)
