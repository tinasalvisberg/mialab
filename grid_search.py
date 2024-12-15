"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from sklearn.model_selection import GridSearchCV
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

LOADING_KEYS_PRE = [structure.BrainImageTypes.T1w,
                    structure.BrainImageTypes.T2w,
                    structure.BrainImageTypes.GroundTruth,
                    structure.BrainImageTypes.BrainMask]


def main(result_dir: str, preprocess_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    pre_process_params = {'training': True,
                          'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'load_images_pre': [False, r'C:\Users\tinas\PycharmProjects\mialab\mia-preprocessed\2024-12-02-21-30-15'],
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True,
                          # Use for ROI-based extraction
                          'use_region_labels': False,
                          # GLCM
                          'texture_contrast_feature': False,
                          'texture_entropy_feature': False,
                          # GLRLM
                          'texture_rlnu_feature': False
                          }

    if pre_process_params['load_images_pre'][0] is False:
        # create a preprocessing directory with timestamp
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        preprocess_dir = os.path.join(preprocess_dir, t)
        os.makedirs(preprocess_dir, exist_ok=True)
        pre_process_params['load_images_pre'][1] = preprocess_dir
        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(data_train_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())
    else:
        crawler = futil.FileSystemDataCrawler(os.path.join(pre_process_params['load_images_pre'][1], 'train'),
                                              LOADING_KEYS_PRE,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter(),
                                              file_extension='.mha')

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # warnings.warn('Random forest parameters not properly set.')
    forest = sk_ensemble.RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [12, 24, 36, 48],
        'max_features': [images[0].feature_matrix[0].shape[1], 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=forest,
                               param_grid=param_grid,
                               cv=5)

    start_time = timeit.default_timer()
    grid_search.fit(data_train, labels_train)

    # Display best parameters and the corresponding score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    best_forest = grid_search.best_estimator_

    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')
    pre_process_params['training'] = False
    # initialize evaluator
    evaluator = putil.init_evaluator()


    # crawl the test image directories
    if pre_process_params['load_images_pre'][0] is False:
        crawler = futil.FileSystemDataCrawler(data_test_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())
    else:
        crawler = futil.FileSystemDataCrawler(os.path.join(pre_process_params['load_images_pre'][1], 'test'),
                                              LOADING_KEYS_PRE,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter(),
                                              file_extension='.mha')

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions = best_forest.predict(img.feature_matrix[0])
        probabilities = best_forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--preprocess_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-preprocessed')),
        help='Directory for preprocessed images.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mialab/data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mialab/data/train/')),
        help='Directory with training data.',
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mialab/data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.preprocess_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)