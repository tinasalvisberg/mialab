# MIA Lab Group 6
Welcome to the project group 6 repository of the autumn semester 2024 MIA Lab course 👋

This repository has been forked from the [mialab repository](https://github.com/ubern-mialab/mialab).

## Focus topic: feature extraction 💡
Is less more when it comes to feature extraction?

**Hypothesis:** a smaller, curated set of features can improve segmentation, compared to using a larger, unfiltered set.

To test the hypothesis, radiomics features are extracted using [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/index.html). The full project report is available as a PDF inside this repository.

## Installation 🔨
Set up an environment and install the libraries listed in `requirements.txt`.

Test the installation by running

    python .\test\test_pyradiomics.py

## Pipeline settings ⚙️
In the file `pipeline.py` you will find the preprocessing parameters in a dictionary called `pre_process_params`. The following settings can be made:

### Saving and loading images after pre-preprocessing


## Run the code 🏃‍♀️
Run the file `pipeline.py`

## Plot results 📊
Create plots from results

    python plot_results.py MIA_RESULTS_FOLDER_NAME

Currently, this plots the results without post-processing as no post-processing is implemented. The file `plot_results.py` would need to be adapted to plot the post-processed results.
