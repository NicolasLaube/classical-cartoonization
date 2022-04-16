"""Main file"""
from tqdm import tqdm

from src.base.image_array import ImageFormat
from src.combiners import HistogramMatcherCombiner, ImageBlenderCombiner, MaskCombiner
from src.dataset.image_dataset import Dataset, DatasetMode, DatasetType
from src.pipelines.chains import CombinerChain, TransformerChain
from src.pipelines.pipeline_transformer import PipelineTransformer
from src.transformers import (
    AdaptiveThresholdContour,
    BinsQuantizationTransformer,
    ColorTransformer,
    GaussianBlurTransformer,
    SuperPixelMode,
    SuperPixelsTransformer,
    TransformerEyesWidener,
    UnskewTransformer,
)

if __name__ == "__main__":

    super_pixel_transformer = TransformerChain(
        name="super_pixel",
        input_name="input",
        output_name="super_pixel",
        transformers=[SuperPixelsTransformer(SuperPixelMode.SLIC)],
    )

    combiner_cartoon = CombinerChain(
        name="cartoon",
        input_name1="input",
        input_name2="super_pixel",
        output_name="output",
        combiner=ImageBlenderCombiner(
            weight_image_1=0.7, weight_image_2=0.3, size=(256, 256)
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

    transformer_eyes_widener = TransformerChain(
        name="transformer_eyes_widener",
        input_name="input",
        output_name="output",
        transformers=[TransformerEyesWidener()],
    )

    transformer_eyes_widener_cart = TransformerChain(
        name="transformer_eyes_widener",
        input_name="input",
        output_name="output",
        transformers=[TransformerEyesWidener()],
    )

    # pipeline_cartoon = PipelineTransformer([transformer_quantized_colors
    # , transformer_chain_contours, combiner_cartoon])
    pipeline_cartoon = PipelineTransformer([transformer_eyes_widener_cart])

    cartoon_dataset = Dataset(DatasetType.CARTOONS, DatasetMode.TRAIN, size=None)

    image_dataset = Dataset(DatasetType.FLICKR, DatasetMode.TRAIN, size=20)

    for image in tqdm(image_dataset, total=len(image_dataset)):
        image_cartoon = pipeline_cartoon(image)
        # plt.imshow(image_cartoon)
        # show_image(image)
        # show_image(cartoon)
        # show_image(image_cartoon)
