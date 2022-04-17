# Cartoonizing with classical computer vision techniques

## Report

You can find the report at this address: `https://drive.google.com/drive/folders/1QZEOePZYLy5r-v1NzK5gERmB5CrUQt3P?usp=sharing`.

## Data

All data is available at `https://drive.google.com/drive/folders/1Ty-eD8pdJNja5iQjBSf8-8xe43IWF0ij?usp=sharing`.

## Weights

Style transfer weights: `https://drive.google.com/drive/folders/1Ty-eD8pdJNja5iQjBSf8-8xe43IWF0ij`.

## Installations

To install dependencies please run `make install`.

## Run pipeline

To run the pipeline please run `make run`.

### Rank cartoons

We developed a tab in the demonstrator to evaluate the results of the cartoonizer. Thanks to this tab, you'll be able to rank the different predictions to estimate which method giuves the best results.
Cartoonized images must be placed in the `data/results/<model_id>` where `<model_id>` is the folder containing all cartonnized images. All predicted images must have the same name as the original image. For example, for the test image `667626_18933d713e.jpg` in flickr dataset should have the same name in the `data\results\model_1` folder.
