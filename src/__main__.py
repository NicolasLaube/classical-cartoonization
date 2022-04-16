"""Main file"""
from tqdm import tqdm

from src.combiners import HistogramMatcherCombiner, ImageBlenderCombiner
from src.dataset.image_dataset import Dataset, DatasetMode, DatasetType
from src.pipelines.chains import CombinerChain, TransformerChain
from src.pipelines.pipeline_transformer import PipelineTransformer
from src.transformers import (
    SuperPixelMode,
    TransformerEyesWidener,
    TransformerSimilarCartoon,
    TransformerSuperPixel,
)

if __name__ == "__main__":

    transformer_similar_cartoon = TransformerChain(
        name="similar_cartoon",
        input_name="input",
        output_name="cartoon",
        transformers=[TransformerSimilarCartoon(plot=False)],
    )

    transformer_super_pixel = TransformerChain(
        name="quick",
        input_name="colored_combined",
        output_name="superpixels",
        transformers=[
            TransformerSuperPixel(SuperPixelMode.QUICKSHIFT, add_rag_thresholding=True)
        ],
    )

    combiner_histogram_matcher = CombinerChain(
        name="combiner_histogram_matcher",
        input_name1="input",
        input_name2="cartoon",
        output_name="colored_combined",
        combiner=HistogramMatcherCombiner(plot=False, colors="rgb"),
    )

    # transformer_quantized_colors = TransformerChain(
    #     name="transformer_quantized_colors",
    #     input_name="input",
    #     output_name="quantized_colors",
    #     transformers=[
    #         TransformerBinsQuantization(n_colors=10),
    #         TransformerGaussianBlur(kernel=(7, 7)),
    #         # HistogramMatchingTransformer(),
    #     ],
    # )
    # transformer_chain_contours = TransformerChain(
    #     name="transformer_chain_contours",
    #     input_name="input",
    #     output_name="contours",
    #     transformers=[
    #         TransformerColor(to_format=ImageFormat.BLACK_AND_WHITE),
    #         TransformerGaussianBlur(kernel=(7, 7)),
    #         TransformerAdaptiveThresholdContour(),
    #         TransformerColor(to_format=ImageFormat.COLORED_WITH_TRANSPARENCY),
    #     ],
    # )

    combiner_blender = CombinerChain(
        name="blender",
        input_name1="colored_combined",
        input_name2="superpixels",
        output_name="combined_superpixels_colors",
        combiner=ImageBlenderCombiner(weight_image_1=0.7, weight_image_2=0.3),
    )

    # combiner_quantized_colors = CombinerChain(
    #     name="combiner_cartoon",
    #     input_name1="quantized_colors",
    #     input_name2="contours",
    #     output_name="output",
    #     combiner=MaskCombiner(opaqueness=1),
    # )

    transformer_eyes_widener = TransformerChain(
        name="transformer_eyes_widener",
        input_name="combined_superpixels_colors",
        output_name="output",
        transformers=[TransformerEyesWidener(plot=False)],
    )

    # transformer_kmeans = TransformerChain(
    #     name="transformer_kmeans",
    #     input_name="input",
    #     output_name="output",
    #     transformers=[
    #         TransformerKMeans(n_colors=25),
    #     ],
    # )

    # pipeline_cartoon = PipelineTransformer([transformer_quantized_colors
    # , transformer_chain_contours, combiner_cartoon])
    pipeline_cartoon = PipelineTransformer(
        [
            transformer_similar_cartoon,
            combiner_histogram_matcher,
            transformer_super_pixel,
            combiner_blender,
            transformer_eyes_widener,
        ]
    )

    cartoon_dataset = Dataset(DatasetType.CARTOONS, DatasetMode.TRAIN, size=None)

    image_dataset = Dataset(
        DatasetType.FLICKR,
        DatasetMode.VALIDATION,
    )

    for i, image in tqdm(enumerate(image_dataset), total=len(image_dataset)):
        image_cartoon = pipeline_cartoon(image, save_path=f"{i}.png")
        # show_image(image_cartoon)
        # plt.imshow(image_cartoon)
        # show_image(image)
        # show_image(cartoon)
        # show_image(image_cartoon)
