
import os

import numpy as np
import pandas as pd

import config
from src.text.classifiers import MLClassifier, TFbertClassifier
from src.image.classifiers import ImgClassifier
from src.multimodal.classifiers import MetaClassifier, TFmultiClassifier
from src.utils.load import load_classifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold


def fit_save_all(params_list, X_train, y_train, X_test, y_test, result_file_name='results.csv'):
    """ 
    The fit_save_all function is designed to automate the process of fitting multiple machine learning models, 
    evaluating their performance, and saving the results along with the models themselves. This function takes 
    a list of parameters for different models, training and testing datasets, and an optional file name for 
    saving the results. It supports multiple classifier types, including ML-based classifiers, 
    BERT-based classifiers for text, and image classifiers. Summary of the results will be appended to the 
    result.csv file if it already exists.

    * Parameters
        * params_list (list of dictionaries): Each dictionary contains the configuration for a model to be trained. 
          Keys include 'modality', 'class', 'base_name', 'vec_method', 'param_grid', and 'nfolds_grid', among others. 
          These dictionaries specify the model type, vectorization method, parameters for grid search, and the number 
          of folds for cross-validation.
        * X_train (DataFrame): The training features dataset.
        * y_train (DataFrame): The training labels dataset.
        * X_test (DataFrame): The testing features dataset.
        * y_test (DataFrame): The testing labels dataset.
        * result_file_name (str, optional): The name of the CSV file to store the results of the model training and 
          evaluation. Defaults to 'results.csv'.

    * Functionality
        Directory and File Preparation: The function first checks if the directory for storing results exists; 
        if not, it creates the directory. It then checks if the specified results CSV file exists within this directory; 
        if not, it creates a new CSV file with the necessary columns.

        Data Preparation for Cross-Validation: It concatenates the training and testing datasets to prepare for 
        cross-validation.

        Model Fitting and Evaluation:
        For each set of parameters in params_list, the function initializes the specified classifier, performs grid 
        search cross-validation if specified, and fits the model on the training data.
        It calculates the F1 score on the test dataset, as well as cross-validated scores on the combined dataset, if applicable.
        Result Recording:

        Saves the best parameters found (if grid search is applied), test scores, cross-validation scores, and the time 
        taken for fitting during cross-validation.
        Saves the trained model to a specified path.
        Updates the results CSV file with the new results for each model.


    * Output
        Returns the results dataframe

    Usage Notes
    Ensure all required libraries (os, pandas, sklearn, etc.) and configurations (e.g., config.path_to_results) 
    are properly set up before using this function.
    This function is highly flexible and supports various classifier types and parameter configurations. Users should 
    carefully prepare the params_list according to their specific requirements for model training and evaluation.
    The function assumes that the models and scoring methods (like classification_score and cross_validate) are implemented 
    and available for use.
    """
    if not os.path.exists(config.path_to_results):
        os.makedirs(config.path_to_results)

    results_path = os.path.join(config.path_to_results, result_file_name)

    # If results.csv doesn't exist, we create it
    if not os.path.isfile(results_path):
        df_results = pd.DataFrame(columns=['modality', 'class', 'vectorization', 'meta_method', 'classifier', 'tested_params',
                                           'best_params', 'score_test', 'score_test_cat', 'conf_mat_test',
                                           'score_train', 'fit_time', 'score_cv_test', 'score_cv_train',
                                           'fit_cv_time', 'probs_test', 'pred_test', 'y_test', 'model_path'])
        df_results.to_csv(results_path)

    # Concatenating train and text sets for CV scores
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    # Mandatory fields in parameters
    mandatory_fields = ['modality', 'class', 'base_name',
                        'vec_method', 'param_grid', 'meta_method', 'model_suffix']

    for params in params_list:
        # Checking mandatory fields
        for fname in mandatory_fields:
            if fname not in params.keys():
                params[fname] = None

        # Making sure params['params_grid'] dictionnary values
        # are provided as list (important for GridSearchCV)
        for key in params['param_grid'].keys():
            if not isinstance(params['param_grid'][key], list):
                params['param_grid'][key] = [params['param_grid'][key]]

        # Populating results with parameters
        results = {'modality': params['modality'], 'class': params['class'], 'classifier': params['base_name'],
                   'vectorization': params['vec_method'], 'meta_method': params['meta_method']}
        # copy the dict params['param_grid'], otherwise with = we copy a reference
        # to the object
        results['tested_params'] = dict(params['param_grid'])

        # Removing validation_data from results dictionnary
        results['tested_params'].pop('validation_data', None)

        # GridsearCV on one parameter
        print('Fitting: ', params['base_name'], params['vec_method'])

        # Fetching first params of list in param_grid in case no GridSearchCV
        # is requested
        clf_params = {}
        for key in params['param_grid'].keys():
            clf_params[key] = params['param_grid'][key][0]

        # Instanciating the classifier
        if params['class'] == 'MLClassifier':
            clf = MLClassifier(
                base_name=params['base_name'], vec_method=params['vec_method'], **clf_params)
        elif params['class'] == 'TFbertClassifier':
            clf = TFbertClassifier(base_name=params['base_name'], **clf_params)
        elif params['class'] == 'ImgClassifier':
            clf = ImgClassifier(base_name=params['base_name'], **clf_params)
        elif params['class'] == 'MetaClassifier':
            estimators_name_list = params['base_name'].split()
            base_estimators = []
            # loading base estimators
            for estimator_name in estimators_name_list:
                clf_base = load_classifier(name=estimator_name)
                if isinstance(clf_base, (TFbertClassifier, ImgClassifier, TFmultiClassifier)):
                    # To make sure copies of this estimator are instanciated with pre-trained weights
                    clf_base.from_trained = estimator_name
                    # epoch = 0 so that the deep networks are not re-trained during cv
                    clf_base.epochs = 0
                base_estimators.append((estimator_name, clf_base))
            clf = MetaClassifier(base_estimators=base_estimators,
                                 meta_method=params['meta_method'], **clf_params)
        elif params['class'] == 'TFmultiClassifier':
            estimators_name_list = params['base_name'].split()
            clf = TFmultiClassifier(
                txt_base_name=estimators_name_list[0], img_base_name=estimators_name_list[1], **clf_params)

        # paramters to feed into GridSearchCV
        param_grid = params['param_grid']

        # Kfold stratification
        if params['nfolds_grid'] > 0:
            cvsplitter = StratifiedKFold(
                n_splits=params['nfolds_grid'], shuffle=True, random_state=123)
        else:
            cvsplitter = None

        # Gridsearch or fit on train set
        if cvsplitter is not None:
            gridcv = GridSearchCV(
                estimator=clf, param_grid=param_grid, scoring='f1_weighted', cv=cvsplitter)
            gridcv.fit(X_train, y_train)
            print('GridSearch: ', gridcv.best_params_)

            # saving best params
            results['best_params'] = gridcv.best_params_

            # Removing validation_data from best_params dictionnary
            results['best_params'].pop('validation_data', None)

            # Keeping the best parameter
            clf = gridcv.best_estimator_
        else:
            clf.fit(X_train, y_train)
            results['best_params'] = None

        results['fit_time'] = clf.fit_time

        # Calculating scores on train set
        f1score_train = clf.classification_score(X_train, y_train)

        # saving f1score_test
        results['score_train'] = f1score_train

        # Calculating scores on test set
        f1score_test = clf.classification_score(X_test, y_test)
        print('Test set, f1score: ', f1score_test)

        # saving f1score_test
        results['score_test'] = f1score_test

        # Computing f1-score for each category from confusion matrix
        recall_mat = clf.confusion_mat / clf.confusion_mat.sum(axis=1)
        precision_mat = clf.confusion_mat / clf.confusion_mat.sum(axis=0)
        f1score_test_cat = 2 * \
            np.diag(precision_mat * recall_mat) / \
            np.diag(precision_mat + recall_mat)
        results['score_test_cat'] = f1score_test_cat.tolist()

        # Saving the confusion matrix
        results['conf_mat_test'] = clf.confusion_mat.tolist()

        # Saving logits and predicted labels
        if hasattr(clf, 'predict_proba'):  # not all objects have predict_proba method
            probs = clf.predict_proba(X_test)
            pred = np.argmax(probs, axis=1)
            results['probs_test'] = probs.tolist()
        else:
            probs = None
            pred = clf.predict(X_test)
            results['probs_test'] = None

        results['pred_test'] = pred.tolist()
        results['y_test'] = y_test.tolist()

        # Calculating score by k-fold cross-validation
        if params['nfolds_cv'] > 0:
            f1score_cv = clf.cross_validate(X, y, cv=params['nfolds_cv'])
            print('CV f1score: ', f1score_cv)

            # saving CV f1score on test, train and fit time
            results['score_cv_test'] = clf.cv_scores['test_score']
            results['score_cv_train'] = clf.cv_scores['train_score']
            results['fit_cv_time'] = clf.cv_scores['fit_time']
        else:
            results['score_cv_test'] = None
            results['score_cv_train'] = None
            results['fit_cv_time'] = None

        # Saving the model (trained on training set only)
        model_name = params['base_name'].replace("/", "-")
        model_name = model_name.replace(" ", "-")

        if params['vec_method'] is not None:
            model_name = model_name + '_' + params['vec_method']
        if params['meta_method'] is not None:
            model_name = params['meta_method'] + '_' + model_name
        if params['model_suffix'] is not None:
            model_name = model_name + '_' + params['model_suffix']

        model_path = params['modality'] + '/' + model_name
        clf.save(model_path)

        # saving where the model is saved
        results['model_path'] = model_path

        # Loading results.csv, adding line and saving it
        # Loading results.csv
        df_results = pd.read_csv(results_path, index_col=0)
        # Adding fields of df_results not in results dictionnary
        for col in df_results.columns:
            if col not in results.keys():
                results[col] = None
        # Adding keys of results dictionnary not in df_results columns
        for col in results.keys():
            if col not in df_results.columns:
                df_results[col] = None

        # Concatenating df_results and results
        # (can't be done by converting results to DataFrame
        # because it contains dictionnaries)
        df_results.loc[len(df_results)] = results
        df_results.to_csv(results_path)

    return df_results
