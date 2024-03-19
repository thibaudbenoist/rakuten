from joblib import load, dump
import os
import notebook.config as config

import ast
import pandas as pd
import numpy as np

import tensorflow as tf

from src.text.classifiers import TFbertClassifier, MLClassifier
from src.image.classifiers import ImgClassifier
from src.multimodal.classifiers import TFmultiClassifier, MetaClassifier


def load_classifier(name, parallel_gpu=False):
    """
    Loads a model from the directory specified in notebook.config file (config.path_to_models).

    Arguments:
    * name: The name of the saved model to load.
    * parallel_gpu: Flag to indicate whether to initialize the model 
        for parallel GPU usage.
    """
    # path to the directory where the model to load was saved
    model_path = os.path.join(config.path_to_models, 'trained_models', name)

    if os.path.isfile(os.path.join(model_path, 'model.joblib')) == False:
        raise ValueError(
            f"The model {name} does not exist in {model_path}. Please check the name and the path to the model. See README.md for details to download model.")

    # Loading the model from there
    loaded_model = load(os.path.join(model_path, 'model.joblib'))

    if isinstance(loaded_model, (TFbertClassifier, ImgClassifier, TFmultiClassifier)):
        # tf.distribute.MirroredStrategy is not saved by joblib
        # so we need to update it here
        if parallel_gpu:
            loaded_model.strategy = tf.distribute.MirroredStrategy()
        else:
            loaded_model.strategy = tf.distribute.OneDeviceStrategy(
                device="/gpu:0")

        # Re-building the model and loading the weights which has been saved
        # in model_path
        if isinstance(loaded_model, TFbertClassifier):
            loaded_model.model, loaded_model.tokenizer = loaded_model._getmodel(
                name)
        elif isinstance(loaded_model, ImgClassifier):
            loaded_model.model, loaded_model.preprocessing_function = loaded_model._getmodel(
                name)
        elif isinstance(loaded_model, TFmultiClassifier):
            loaded_model.model, loaded_model.tokenizer, loaded_model.preprocessing_function = loaded_model._getmodel(
                name)

        if hasattr(loaded_model, 'callbacks'):
            if loaded_model.callbacks is not None:
                callbacks = loaded_model.callbacks
                loaded_model.callbacks = [
                    callback for callback in callbacks if callback[0] != 'TensorBoard']

        # For copies of loaded object to also inherit from name (e.g.
        # for ensemble methods or CV methods), we set from_trained
        # to the model loaded here
        loaded_model.from_trained = name

    if isinstance(loaded_model, MetaClassifier):
        # Loading all base estimators from the  subfolders. These should go into
        # loaded_model.base_estimator, loaded_model.model.estimators_ ,
        # loaded_model.model.estimators and loaded_model.model.named_estimators_
        for k, clf in enumerate(loaded_model.base_estimators):
            # loading the base estimators models
            base_model = load_classifier(name + os.sep + clf[0])

            # Casting them into the necessary attributes
            # in base_estimators
            loaded_model.base_estimators[k] = (clf[0], base_model)
            # in model.estimators_
            if isinstance(loaded_model.model.estimators_[0], tuple):
                loaded_model.model.estimators_[k] = (
                    loaded_model.model.estimators_[k][0], base_model)
            else:
                loaded_model.model.estimators_[k] = base_model
            # in model.estimators
            if isinstance(loaded_model.model.estimators[0], tuple):
                loaded_model.model.estimators[k] = (
                    loaded_model.model.estimators[k][0], base_model)
            # in model.named_estimators_
            keyname = list(loaded_model.model.named_estimators_.keys())[k]
            loaded_model.model.named_estimators_[keyname] = base_model

    return loaded_model


def load_batch_results(filename):
    """ 
    Load batch results saved in filename in the directory specified in
    config.path_to_results.

    Example:
    df = load_batch_results('results_benchmark_text')
    """
    df_results = pd.read_csv(os.path.join(
        config.path_to_results, filename + '.csv'), index_col=0)

    col_to_convert = ['conf_mat_test', 'probs_test',
                      'pred_test', 'y_test', 'score_test_cat']
    for col in col_to_convert:
        df_results.loc[~df_results[col].isna(), col] = df_results.loc[~df_results[col].isna(
        ), col].apply(ast.literal_eval).apply(np.array)

    return df_results


def fix_results(result_filename, X_test, y_test):
    """ 
    To be strip later. Just a temporary fix for batch performed before 2024/03/12
    """
    df_results = pd.read_csv(os.path.join(
        config.path_to_results, result_filename + '.csv'), index_col=0)

    col_to_add = ['score_test_cat', 'conf_mat_test', 'score_train',
                  'fit_time', 'probs_test', 'pred_test', 'y_test']
    for col in col_to_add:
        if col not in df_results.columns:
            df_results[col] = None

    df_results = df_results[['modality', 'class', 'vectorization', 'meta_method', 'classifier', 'tested_params', 'best_params', 'score_test', 'score_test_cat', 'conf_mat_test', 'score_train', 'fit_time',
                            'score_cv_test', 'score_cv_train', 'fit_cv_time', 'probs_test', 'pred_test', 'y_test', 'model_path']]

    for i in df_results.index:
        clf = load_classifier(df_results.loc[i, 'model_path'])
        df_results.loc[i, 'fit_time'] = clf.fit_time

        if isinstance(clf.confusion_mat, pd.DataFrame):
            clf.confusion_mat = np.array(clf.confusion_mat)
            clf.save(df_results.loc[i, 'model_path'])

        df_results.loc[i, 'conf_mat_test'] = str(clf.confusion_mat.tolist())

        recall_mat = clf.confusion_mat / clf.confusion_mat.sum(axis=1)
        precision_mat = clf.confusion_mat / clf.confusion_mat.sum(axis=0)
        f1score_test_cat = 2 * \
            np.diag(precision_mat * recall_mat) / \
            np.diag(precision_mat + recall_mat)
        df_results.loc[i, 'score_test_cat'] = str(f1score_test_cat.tolist())

        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_test)
            pred = np.argmax(probs, axis=1)
            df_results.loc[i, 'probs_test'] = str(probs.tolist())
        else:
            probs = None
            pred = clf.predict(X_test)
            df_results.loc[i, 'probs_test'] = None

        df_results.loc[i, 'pred_test'] = str(pred.tolist())
        df_results.loc[i, 'y_test'] = str(y_test.tolist())

    df_results = df_results.sort_values(
        by='score_test', ascending=False).reset_index(drop=True)

    return df_results
