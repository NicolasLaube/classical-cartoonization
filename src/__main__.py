"""Main file"""
from src.base.image_array import ImageFormat
from src.combiners import HistogramMatcherCombiner, ImageBlenderCombiner, MaskCombiner
from src.dataset.image_dataset import Dataset, DatasetMode, DatasetType
from src.dataset.utils import show_image
from src.pipelines.chains import CombinerChain, TransformerChain
from src.pipelines.pipeline_transformer import PipelineTransformer
from src.transformers import (
    AdaptiveThresholdContour,
    BinsQuantizationTransformer,
    ColorTransformer,
    GaussianBlurTransformer,
    UnskewTransformer,
)

if __name__ == "__main__":
    combiner_cartoon = CombinerChain(
        name="cartoon",
        input_name1="cartoon",
        input_name2="input",
        output_name="output",
        combiner=ImageBlenderCombiner(
            weight_image_1=0.5, weight_image_2=0.5, size=(256, 256)
        ),
    )

    transformer_unskew = TransformerChain(
        name="unskew",
        input_name="input",
        output_name="output",
        transformers=[UnskewTransformer()],
    )

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
    combiner_quantized_colors = CombinerChain(
        name="combiner_cartoon",
        input_name1="quantized_colors",
        input_name2="contours",
        output_name="output",
        combiner=MaskCombiner(opaqueness=1),
    )

    combiner_histogram_matcher = CombinerChain(
        name="combiner_histogram_matcher",
        input_name1="input",
        input_name2="cartoon",
        output_name="output",
        combiner=HistogramMatcherCombiner(),
    )

    # pipeline_cartoon = PipelineTransformer([transformer_quantized_colors
    # , transformer_chain_contours, combiner_cartoon])
    pipeline_cartoon = PipelineTransformer(
        [
            combiner_histogram_matcher
            # combiner_cartoon
        ]  # transformer_quantized_colors, transformer_chain_contours, combiner_cartoon
    )

    cartoon_dataset = Dataset(DatasetType.CARTOONS, DatasetMode.TRAIN, size=20)

    image_dataset = Dataset(DatasetType.FLICKR, DatasetMode.TRAIN, size=20)

    for image, cartoon in zip(image_dataset, cartoon_dataset):
        image_cartoon = pipeline_cartoon(image, cartoon)
        # plt.imshow(image_cartoon)
        show_image(image)
        show_image(cartoon)
        show_image(image_cartoon)
