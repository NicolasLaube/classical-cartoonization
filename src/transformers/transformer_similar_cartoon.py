"""Transformer Similar Cartoon."""
import os

from src.base.image_array import ImageArray
from src.dataset.utils import read_image, show_image
from src.models.predictor_similar_cartoons import PredictorSimilarCartoon


class TransformerSimilarCartoon:
    """Transformer Similar Cartoon."""

    def __init__(self, plot: bool = True) -> None:
        self.similar_cartoon_predictor = PredictorSimilarCartoon()
        self.plot = plot

    def __call__(self, image: ImageArray) -> ImageArray:
        """Apply super pixels."""
        path, _ = self.similar_cartoon_predictor.get_most_similar_image_array(image)
        cartoon = read_image(
            os.path.abspath(
                path.replace("cartoon_features", "cartoon_frames").replace("npy", "jpg")
            ),
            add_base=False,
        )
        if cartoon is not None:
            if self.plot:
                self.show(cartoon)
            return cartoon
        raise Exception("Could not find cartoon")

    @staticmethod
    def show(cartoon: ImageArray) -> None:
        """Show transformer."""

        show_image(cartoon)
