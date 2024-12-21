# MIA Lab Group 6
Welcome to the project group 6 repository of the autumn semester 2024 MIA Lab course 👋

This repository has been forked from the [mialab repository](https://github.com/ubern-mialab/mialab).

## Focus topic: feature extraction 💡
Is less more when it comes to feature extraction?

**Hypothesis:** a smaller, curated set of features can improve segmentation, compared to using a larger, unfiltered set.

To test the hypothesis, radiomics features are extracted using [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/index.html). The full project report is available as a PDF inside this repository.

## Installation 🔨
Set up an environment with Python version 3.10 and install the libraries listed in `requirements.txt`. We have not tested the pipeline with other Python versions, but it might work.

Test the installation by running

    python .\test\test_pyradiomics.py

## Pipeline settings ⚙️
In the file `pipeline.py` you will find the preprocessing parameters in a dictionary called `pre_process_params`. The following settings can be made:

### Saving and loading images after pre-preprocessing
When `load_images_pre` is set to `False` the whole pipeline will run as usual and the pre-processed images will be saved in a time-stamped folder in the `mia-preprocessed` folder.

When you want to save time during execution or make the results as reproducible as possible you can load the saved pre-processed images from a folder. Set `load_images_pre` to `True` and indicate the path to the folder that you want the images to load from. The steps skull-stripping, normalisation and registration will then automatically be skipped and you do not need to deactivate them manually.

### Selecting features
The features for the feature extraction can be toggled individually. From the original pipeline:
* Coordinates: `coordinates_feature`
* Intensity: `intensity_feature`
* Gradient intensity: `gradient_intensity_feature`

Features from PyRadiomics:
* Gray Level Co-occurrence Matrix Contrast: `texture_contrast_feature`
* Gray Level Co-occurrence Matrix Difference Entropy: `texture_entropy_feature`
* Gray Level Run Length Matrix Run Length Non-Uniformity: `texture_rlnu_feature`

## Run the code 🏃‍♀️
Run the file `pipeline.py`.

## Plot results 📊
The following command creates plots from the results:

    python plot_results.py MIA_RESULTS_FOLDER_NAME

Currently, this plots the results without post-processing as no post-processing is implemented. The file `plot_results.py` would need to be adapted to plot the post-processed results.

The result overview plots used in the report were generated with the MatLab script `heat_maps.m`.
