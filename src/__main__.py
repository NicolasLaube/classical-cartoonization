"""Main file"""
from src.base.image_array import ImageFormat
from src.combiners import MaskCombiner
from src.dataset.image_dataset import Dataset, DatasetMode, DatasetType
from src.dataset.utils import show_image
from src.pipelines.chains import CombinerChain, TransformerChain
from src.pipelines.pipeline_transformer import PipelineTransformer
from src.transformers import (
    AdaptiveThresholdContour,
    BinsQuantizationTransformer,
    ColorTransformer,
    GaussianBlurTransformer,
)

if __name__ == "__main__":
    transformer_quantized_colors = TransformerChain(
        name="transformer_quantized_colors",
        input_name="input",
        output_name="quantized_colors",
        transformers=[
            BinsQuantizationTransformer(n_colors=10),
            GaussianBlurTransformer(kernel=(7, 7)),
            # HistogramMatchingTransformer(),
        ],
    )
    transformer_chain_contours = TransformerChain(
        name="transformer_chain_contours",
        input_name="input",
        output_name="contours",
        # transformers=[ColorTransformer(to_format=ImageFormat.BLACK_AND_WHITE),
        #  BlurTransformer(kernel=(7,7)), CannyEdgeTransformer(), DilateErodeTransformer(),
        #  BlurTransformer(), ContourTransformer(),
        # ColorTransformer(to_format=ImageFormat.COLORED_WITH_TRANSPARENCY)]
        transformers=[
            ColorTransformer(to_format=ImageFormat.BLACK_AND_WHITE),
            GaussianBlurTransformer(kernel=(7, 7)),
            AdaptiveThresholdContour(),
            ColorTransformer(to_format=ImageFormat.COLORED_WITH_TRANSPARENCY),
        ],
    )
    combiner_cartoon = CombinerChain(
        name="combiner_cartoon",
        input_name1="quantized_colors",
        input_name2="contours",
        output_name="output",
        combiner=MaskCombiner(opaqueness=1),
    )

    # pipeline_cartoon = PipelineTransformer([transformer_quantized_colors
    # , transformer_chain_contours, combiner_cartoon])
    pipeline_cartoon = PipelineTransformer(
        [transformer_quantized_colors, transformer_chain_contours, combiner_cartoon]
    )

    image_dataset = Dataset(DatasetType.FLICKR, DatasetMode.TRAIN)

    for image in image_dataset:
        image_cartoon = pipeline_cartoon(image)
        show_image(image_cartoon)
