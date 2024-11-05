# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to get started with classical medical image analysis.

During the MIALab you will work on the task of brain tissue segmentation from magnetic resonance (MR) images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification
- Post-processing
- Evaluation

After you complete the exercises, dive into the 
    
    pipeline.py 

script to learn how all of these steps work together. 

During the laboratory you will get to know the entire pipeline and investigate one of these pipeline elements in-depth.
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer or researcher in the medical image analysis domain.

Enjoy!

----
## Group 6: Is less more when it comes to feature extraction? 

**Hypothesis:** Using radiomics features on top of raw image data improves the segmentation result.

### Installation
Make sure to install [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/index.html) using `requirement.txt`:

    python -m pip install pyradiomics

Test installation with

    python .\test\test_pyradiomics.py

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.
