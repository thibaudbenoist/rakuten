from importlib import reload
import pandas as pd
import src.utils.plot as uplot
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ast
import os
import plotly.express as px
import re
from src.utils.load import load_classifier
from sklearn.metrics import classification_report, f1_score
from tabulate import tabulate
import requests
from PIL import Image
from io import BytesIO
from src.utils.visualize import deepCAM, plot_weighted_text
import time


class ResultsManager():
    """
    Loads and manages the results of the models trained.
    Results are stored in a CSV file with the following columns:
    * modality: the type of data used (e.g. text, img, multimodal)
    * class: the class of the model (e.g. MLClassifier, RandomForestClassifier)
    * vectorization: the type of vectorization used (e.g. TfidfVectorizer, CountVectorizer)
    * classifier: the type of classifier used (e.g. LogisticRegression, RandomForestClassifier)
    * tested_params: the parameters used for the model training
    * best_params: the best parameters found by the grid search
    * score_test: the f1 score on the test set
    * score_test_cat: the f1 score per category on the test set
    * conf_mat_test: the confusion matrix on the test set
    * score_train: the f1 score on the train set
    * fit_time: the time it took to fit the model
    * score_cv_test: the f1 score on the cross validation test set
    * score_cv_train: the f1 score on the cross validation train set
    * fit_cv_time: the time it took to fit the model on the cross validation set
    * probs_test: the probabilities of the predictions on the test set
    * pred_test: the predictions on the test set
    * y_test: the true labels of the test set
    * model_path: the path to the model file
    * package: a classification of the model (e.g. text, img, bert)
    """

    def __init__(self, config) -> None:
        self.config = config
        self.df_results = None
        self.le = None
        self.X_test = None
        self.deepCam = None
        self.loaded_classifiers = {}
        pass

    def add_result_file(self, file_path, package):
        """
        Add a result file to the results manager.
        The file should be a CSV file with the following columns:
        * modality: the type of data used (e.g. text, img, multimodal)
        * class: the class of the model (e.g. MLClassifier, RandomForestClassifier)
        * vectorization: the type of vectorization used (e.g. TfidfVectorizer, CountVectorizer)
        * classifier: the type of classifier used (e.g. LogisticRegression, RandomForestClassifier)
        * tested_params: the parameters used for the model training
        * best_params: the best parameters found by the grid search
        * score_test: the f1 score on the test set
        * score_test_cat: the f1 score per category on the test set
        * conf_mat_test: the confusion matrix on the test set
        * score_train: the f1 score on the train set
        * fit_time: the time it took to fit the model
        * score_cv_test: the f1 score on the cross validation test set
        * score_cv_train: the f1 score on the cross validation train set
        * fit_cv_time: the time it took to fit the model on the cross validation set
        * probs_test: the probabilities of the predictions on the test set
        * pred_test: the predictions on the test set
        * y_test: the true labels of the test set
        * model_path: the path to the model file

        Args:
            file_path (str): the path to the CSV file
            package (str): a classification of the model (e.g. text, img, bert)

        Returns:
            ResultsManager: the results manager
        """
        df = pd.read_csv(file_path, index_col=0)
        df['package'] = package
        if self.df_results is None:
            self.df_results = df
        else:
            self.df_results = pd.concat(
                [self.df_results, df]).reset_index(drop=True)
        self.df_results = self.df_results.drop_duplicates()

        return self

    def plot_f1_scores(self, filter_package=None, filter_model=None, figsize=(1200, 600), title=None):
        """
        Plot the f1 scores of the models on an horizontal bar plot.

        Args:
            filter_package (list, optional): a list of packages to filter the results. Defaults to None.
            filter_model (list, optional): a list of model paths to filter the results. Defaults to None.
            figsize (tuple, optional): the size of the plot. Defaults to (1200, 600).
            title (str, optional): the title of the plot. Defaults to None.

        Returns:
            ResultsManager: the results manager
        """

        fig = self.build_fig_f1_scores(
            filter_package=filter_package,
            filter_model=filter_model,
            figsize=figsize,
            title=title
        )
        fig.show()

        return self

    def build_fig_f1_scores(self, filter_package=None, filter_model=None, figsize=(1200, 600), title=None):
        """
        Instantiate fig object of the f1 scores of the models on an horizontal bar plot.

        Args:
            filter_package (list, optional): a list of packages to filter the results. Defaults to None.
            filter_model (list, optional): a list of model paths to filter the results. Defaults to None.
            figsize (tuple, optional): the size of the plot. Defaults to (1200, 600).
            title (str, optional): the title of the plot. Defaults to None.

        Returns:
            ResultsManager: the results manager
        """

        # filtre des doublons
        scores = self.df_results[[
            'model_path',
            'score_test',
            'package',
            'classifier',
            'vectorization']].reset_index(drop=True)
        index_to_keep = scores.groupby(['model_path'])['score_test'].idxmax()
        scores = scores.loc[index_to_keep].reset_index()

        # filtre des packages
        if filter_package is not None:
            scores = scores[
                (scores.package.isin(filter_package))
            ].reset_index(drop=True)
        if filter_model is not None:
            scores = scores[
                (scores.model_path.isin(filter_model))
            ].reset_index(drop=True)

        # ajout des colonnes pour le plot

        scores['serie_name'] = scores.apply(lambda row: row.model_path.split('/')[-1] if pd.isna(
            row.vectorization) else row.classifier + ' - ' + row.vectorization, axis=1)
        scores['vectorizer'] = scores.apply(lambda row: row.model_path.split('/')[-1] if pd.isna(
            row.vectorization) else row.vectorization, axis=1)

        # tri par score décroissant
        scores = scores[['serie_name', 'score_test',
                         'vectorizer']].reset_index()
        sorted_scores = scores.sort_values(by='score_test', ascending=False)

        # plot
        if title is None:
            title = 'Benchmark des f1 scores'
        fig = uplot.get_fig_benchs_results(
            sorted_scores,
            'serie_name',
            'score_test',
            'model',
            'f1 score',
            color_column='vectorizer',
            title=title,
            figsize=figsize
        )

        return fig

    def plot_f1_scores_by_prdtype(self, filter_package=None, filter_model=None, figsize=(1200, 600), title=None):
        """
        Plot the f1 scores of the models by category on an horizontal bar plot.

        Args:
            filter_package (list, optional): a list of packages to filter the results. Defaults to None.
            filter_model (list, optional): a list of model paths to filter the results. Defaults to None.
            figsize (tuple, optional): the size of the plot. Defaults to (1200, 600).
            title (str, optional): the title of the plot. Defaults to None.

        Returns:
            ResultsManager: the results manager
        """
        scores = self.df_results[[
            'model_path',
            'score_test',
            'package',
            'classifier',
            'vectorization',
            'score_test_cat']].reset_index(drop=True)
        scores = scores[scores.score_test_cat.notna()].reset_index(drop=True)
        index_to_keep = scores.groupby(['model_path'])['score_test'].idxmax()
        scores = scores.loc[index_to_keep].reset_index()

        # filtre des packages
        if filter_package is not None:
            scores = scores[
                (scores.package.isin(filter_package))
            ].reset_index(drop=True)
        if filter_model is not None:
            scores = scores[
                (scores.model_path.isin(filter_model))
            ].reset_index(drop=True)

        # ajout des colonnes pour le plot
        scores['serie_name'] = scores.apply(lambda row: row.classifier if pd.isna(
            row.vectorization) else row.classifier + ' - ' + row.vectorization, axis=1)
        scores['vectorizer'] = scores.apply(lambda row: row.classifier if pd.isna(
            row.vectorization) else row.vectorization, axis=1)

        scores_to_plot = None

        for index, row in scores.iterrows():
            score_to_plot = pd.DataFrame(ast.literal_eval(row.score_test_cat))
            score_to_plot.columns = ['score_test']
            score_to_plot['model'] = row.serie_name
            score_to_plot['category'] = self.get_cat_labels()

            if scores_to_plot is None:
                scores_to_plot = score_to_plot
            else:
                scores_to_plot = pd.concat([scores_to_plot, score_to_plot])
        scores_to_plot = scores_to_plot.sort_values(
            by='score_test', ascending=False)

        fig = px.bar(
            scores_to_plot,
            x='category',
            y='score_test',
            color='model',
            barmode='group',
        )

        fig.update_traces(
            width=0.2,
        )
        # Update layout to remove legend and adjust xaxis title
        fig.update_layout(
            legend=None,
            xaxis_title='f1 score',
            yaxis_title='classe',
            width=1200,
            height=600,
            title=title,
        )

        # Show the plot
        fig.show()
        return self

    def get_cat_labels(self):
        """
        Get the category labels.

        Returns:
            list: the category labels ordered by index
        """
        if self.le is None:
            self.le = LabelEncoder()
            self.le.classes_ = np.load(os.path.join(
                self.config.path_to_data, 'le_classes.npy'), allow_pickle=True)

        return self.le.classes_

    def get_label_encoder(self):
        """
        Get the label encoder.

        Returns:
            LabelEncoder: the label encoder
        """
        if self.le is None:
            self.le = LabelEncoder()
            self.le.classes_ = np.load(os.path.join(
                self.config.path_to_data, 'le_classes.npy'), allow_pickle=True)

        return self.le

    def get_num_classes(self):
        """
        Get the number of classes.

        Returns:
            int: the number of classes
        """
        return len(self.get_cat_labels())

    def plot_classification_report(self, model_path, model_label=None):
        """
        Display the classification report of a model.
        Plot the confusion matrix of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.
        """
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        uplot.classification_results(
            y_test,
            y_pred,
            index=self.get_cat_labels(),
            model_label=model_label
        )
        return self

    def plot_confusion_matrix(self, model_path, model_label=None):
        """
        Plot the confusion matrix of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.

        Returns:
            ResultsManager: the results manager
        """
        fig = self.get_fig_confusion_matrix(model_path, model_label)
        fig.show()

        return self

    def get_fig_confusion_matrix(self, model_path, model_label=None):
        """
        Build the figure of the confusion matrix of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.

        Returns:
            figure to plot
        """
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        return uplot.get_fig_confusion_matrix(
            y_test,
            y_pred,
            index=self.get_cat_labels(),
            model_label=model_label
        )

    def get_fig_compare_confusion_matrix(self, model_path1, model_path2, model_label1=None, model_label2=None):
        y_pred1 = self.get_y_pred(model_path1)
        y_pred2 = self.get_y_pred(model_path2)
        y_test = self.get_y_test(model_path1)

        return uplot.get_fig_compare_confusion_matrix(
            y_test,
            y_pred1,
            y_pred2,
            index=self.get_cat_labels(),
            model_label1=model_label1,
            model_label2=model_label2
        )

    def plot_f1_scores_report(self, model_path, model_label=None):
        """
        Display the classification report of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.
        """
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        print(classification_report(y_test, y_pred,
              target_names=self.get_cat_labels()))

        return self

    def get_f1_scores_report(self, model_path, model_label=None):
        """
        Display the classification report of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.
        """
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        return classification_report(y_test, y_pred,
                                     target_names=self.get_cat_labels(), output_dict=True)

    def plot_classification_report_merged(self, model_paths):
        """
        Display the classification report of further models merged taking the best predictions of all models.

        Args:
            model_paths (list): a list of model paths

        Returns:
            ResultsManager: the results manager
        """
        y_pred = []
        y_test = self.get_y_test(model_paths[0])
        for model_path in model_paths:
            y_pred.append(self.get_y_pred(model_path))

        y_merged = []
        for i, y in enumerate(y_test):
            merged_label = y_pred[0][i]
            for y_p in y_pred:
                if y_p[i] == y:
                    merged_label = y
                    break
            y_merged.append(merged_label)

        uplot.classification_results(
            y_test, y_merged, index=self.get_cat_labels())
        return self

    def get_y_pred(self, model_path):
        """
        Get the predictions of a model.
        If not available in the dataset results, the model is loaded and the predictions are computed.

        Args:
            model_path (str): the path to the model file

        Returns:
            np.array: the predictions
        """
        if pd.isna(self.df_results[self.df_results.model_path == model_path].pred_test.values[0]):
            clf = self.load_classifier(model_path)
            y_pred = clf.predict(self.get_X_test())
            return y_pred
        pred = self.df_results[self.df_results.model_path ==
                               model_path].pred_test.values[0]

        return ast.literal_eval(pred)

    def get_deepCam(self):
        if self.deepCam is None:
            clf_fusion = self.load_classifier(
                'fusion/camembert-base-vit_b16_TF6')
            self.deepCam = deepCAM(clf_fusion)

        return self.deepCam

    def predict(self, models_paths, text=None, img_url=None):

        probas = []
        weight_set = []
        img_array = None
        icam = None

        if img_url:
            img_res = requests.get(img_url, stream=True)
            if img_res.status_code == 200:
                img = Image.open(BytesIO(img_res.content))
                img_array = np.array(img)
        start_time = time.time()
        for basename in models_paths:
            probas.append(
                np.array(self.predict_proba(basename, text, img_array)))
            weight_set.append(self.get_f1_score(basename))

        probas_weighted = np.sum([probas[i] * weight_set[i]
                                  for i in range(len(probas))], axis=0)/np.sum(weight_set)
        end_time = time.time()

        pred = np.argmax(probas_weighted, axis=1)

        labels = self.get_label_encoder().inverse_transform(pred)

        # gradcam

        # Pick some data
        # idx = 0
        # image = cv2.imread(data['img_path'][idx])
        # text = data['tokens'][idx]

        # Format it as a dictionnary since here we've got a fusion model
        X = {'text': text, 'image': img_array}

        # Compute masks and masked inputs
        icam = self.get_deepCam()
        icam.computeMaskedInput(X, min_factor=0.0)
        return {'pred_labels': labels, 'pred_probas': probas_weighted.tolist(), 'labels': self.get_cat_labels(), 'icam': icam, 'time': end_time - start_time}

    def load_classifier(self, model_path):
        """
        Load a classifier from a model path.

        Args:
            model_path (str): the path to the model file

        Returns:
            Classifier: the classifier
        """
        if model_path not in self.loaded_classifiers.keys():
            self.loaded_classifiers[model_path] = load_classifier(model_path)

        return self.loaded_classifiers[model_path]

    def predict_proba(self, model_path, text=None, img=None):
        clf = self.load_classifier(model_path)

        model_type = model_path.split('/')[0]
        if (model_type == 'text' or model_type == 'bert') and text:
            if hasattr(clf, 'predict_proba'):
                probas = clf.predict_proba(text)
            else:
                probas = np.zeros(
                    (1, self.get_num_classes()))
                pred = clf.predict(text)
                probas[0, pred] = 1
        elif model_type == 'image' and not img is None:
            probas = clf.predict_proba(img)
        else:
            probas = clf.predict_proba(
                {'text': text, 'image': img})

        return probas

    def get_y_pred_probas(self, model_path):
        """
        Get the probabilities of the predictions of a model.
        If not available in the dataset results, the model is loaded and the probabilities are computed.
        If the model does not have a predict_proba method, the probabilities are computed from the predictions.

        Args:
            model_path (str): the path to the model file

        Returns:
            np.array: the probabilities
        """
        if pd.isna(self.df_results[self.df_results.model_path == model_path].probs_test.values[0]):
            clf = self.load_classifier(model_path)

            if hasattr(clf, 'predict_proba'):
                probas = clf.predict_proba(self.get_X_test())
            else:
                probas = np.zeros(
                    (len(self.get_X_test()), self.get_num_classes()))
                y_pred = self.get_y_pred(model_path)
                for i, y in enumerate(y_pred):
                    probas[i, y] = 1

            return probas
        return ast.literal_eval(self.df_results[self.df_results.model_path ==
                                                model_path].probs_test.values[0])

    def get_y_test(self, model_path):
        """
        Get the true labels of the test set of a model.

        Args:
            model_path (str): the path to the model file

        Returns:
            np.array: the true labels
        """
        if pd.isna(self.df_results[self.df_results.model_path == model_path].y_test.values[0]):
            df = pd.read_csv(os.path.join(
                self.config.path_to_data, 'df_test_index.csv'))
            return df.prdtypeindex.values

        return ast.literal_eval(self.df_results[self.df_results.model_path == model_path].y_test.values[0])

    def get_X_test(self):
        """
        Get the test set.

        Returns:
            pd.DataFrame: the test set
        """
        if self.X_test is None:
            df = pd.read_csv(os.path.join(
                self.config.path_to_data, 'df_test_index.csv'))
            colnames = ['designation_translated', 'description_translated']
            df['tokens'] = df[colnames].apply(lambda row: ' '.join(
                s.lower() for s in row if isinstance(s, str)), axis=1)
            df['img_path'] = df.apply(lambda row:
                                      os.path.join(self.config.path_to_images, 'image_'
                                                   + str(row['imageid'])
                                                   + '_product_'
                                                   + str(row['productid'])
                                                   + '_resized'
                                                   + '.jpg'),
                                      axis=1)
            self.X_test = df[['tokens', 'img_path']]
        return self.X_test

    def get_model_paths(self, filter_package=None):
        """
        Get all the model paths loaded in the results manager.

        Returns:
            list: the model paths
        """
        if filter_package is not None:
            return self.df_results[self.df_results.package.isin(filter_package)].model_path.unique()

        return self.df_results.model_path.unique()

    def get_model_label(self, model_path):
        """
        Get the label of a model.

        Args:
            model_path (str): the path to the model file

        Returns:
            str: the label of the model
        """
        label = self.df_results[self.df_results.model_path ==
                                model_path].classifier.values[0]
        if not pd.isna(self.df_results[self.df_results.model_path == model_path].vectorization.values[0]):
            label += '(' + self.df_results[self.df_results.model_path ==
                                           model_path].vectorization.values[0] + ')'

        return label

    def get_f1_score(self, model_path):
        """
        Get the f1 score of a model.

        Args:
            model_path (str): the path to the model file

        Returns:
            float: the f1 score
        """
        return self.df_results[self.df_results.model_path == model_path].score_test.values[0]

    def voting_pred(self, basenames):
        """
        Get the predictions of a voting model.
        Computes the weighted average of the predictions of the models.

        Args:
            basenames (list): a list of model paths

        Returns:    
            np.array: the predictions
        """
        probas = []
        weight_set = []

        for basename in basenames:
            probas.append(np.array(self.get_y_pred_probas(basename)))
            weight_set.append(self.get_f1_score(basename))

        probas_weighted = np.sum([probas[i] * weight_set[i]
                                  for i in range(len(probas))], axis=0)
        y_pred = np.argmax(probas_weighted, axis=1)
        return y_pred

    def voting_pred_cross_validate(self, basenames, n_folds=5, dataset_size=0.5):
        """
        Print the f1 score of a voting model with cross validation.
        Computes the weighted average of the predictions of the models.

        Args:
            basenames (list): a list of model paths
            n_folds (int, optional): the number of folds. Defaults to 5.
            dataset_size (float, optional): the size of the dataset. Defaults to 0.5.

        Returns:
            f1_score (float): the f1 mean score on the test set
        """
        f_score_cv = []
        f_score_cv_macro = []
        probas = []
        weight_set = []
        report = []

        for basename in basenames:
            probas.append(np.array(self.get_y_pred_probas(basename)))
            s = self.get_f1_score(basename)
            weight_set.append(s)
            report.append([basename, s])

        y_test = np.array(self.get_y_test(basenames[0]))
        test_idx_start = round((len(y_test) * (1-dataset_size)))
        test_size = len(y_test) - test_idx_start

        for k in range(n_folds):

            idx_start = test_idx_start + round((k-1)*(test_size/n_folds))
            idx_end = test_idx_start + round((k)*(test_size/n_folds) + 1)
            idx_fold = range(idx_start, idx_end)

            train_mask = np.ones(len(y_test), dtype=bool)
            train_mask[:test_idx_start] = False
            train_mask[idx_fold] = False

            test_mask = np.zeros(len(y_test), dtype=bool)
            test_mask[idx_fold] = True

            for basename in basenames:
                probas.append(np.array(self.get_y_pred_probas(basename)))
                y_pred = np.array(self.get_y_pred(basename))
                weight_set.append(
                    f1_score(y_test[train_mask], y_pred[train_mask], average='weighted'))

            probas_weighted = np.sum([probas[i] * weight_set[i]
                                      for i in range(len(probas))], axis=0)
            y_pred = np.argmax(probas_weighted, axis=1)
            f_score_cv.append(
                f1_score(y_test[test_mask], y_pred[test_mask], average='weighted'))
            f_score_cv_macro.append(
                f1_score(y_test[test_mask], y_pred[test_mask], average='macro'))
            report.append(['voting fold ' + str(k), f_score_cv[-1]])

        report.append(['voting mean weighted', np.mean(f_score_cv)])
        report.append(['voting mean macro', np.mean(f_score_cv_macro)])
        print(tabulate(report, headers=[
              'model / fold', 'f1 score']))

        return np.mean(f_score_cv)

    def get_voting_confusion_matrix(self, models_paths, model_label=None):
        y_pred = self.voting_pred(models_paths)
        y_test = self.get_y_test(models_paths[0])

        return uplot.get_fig_confusion_matrix(
            y_test,
            y_pred,
            index=self.get_cat_labels(),
            model_label=model_label
        )

    def get_voting_f1_scores_report(self, models_paths, model_label=None):
        """
        Display the classification report of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.
        """
        y_pred = self.voting_pred(models_paths)
        y_test = self.get_y_test(models_paths[0])

        return classification_report(y_test, y_pred,
                                     target_names=self.get_cat_labels(), output_dict=True)

    def get_false_samples(self, model_path, n_samples=5):
        """
        Build the figure of the confusion matrix of a model.

        Args:
            model_path (str): the path to the model file
            model_label (str, optional): the label of the model displayed on the report. Defaults to None.

        Returns:
            figure to plot
        """
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        y_pred = self.get_label_encoder().inverse_transform(y_pred)
        y_test = self.get_label_encoder().inverse_transform(y_test)

        false_samples = np.equal(y_pred, y_test)
        X_test = self.get_X_test()
        X_test['pred'] = y_pred
        X_test['true'] = y_test
        X_false = X_test[false_samples == False]
        return X_false.head(n_samples)
