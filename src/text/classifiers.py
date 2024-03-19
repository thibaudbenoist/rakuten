"""
Class implementations for constructing and utilizing machine learning models, specifically focusing on BERT models 
for text classification and traditional ML algorithms. Below is a summary documentation of the classes and 
functions along with their parameters and available methods.

* TFbertClassifier Class:
    Implements a TensorFlow-based BERT classifier extending the BaseEstimator and ClassifierMixin from scikit-learn, for seamless integration with scikit-learn utilities.

    Constructor Parameters:
        * base_name: Identifier for the pre-trained base BERT model.
        * from_trained (optional): Name of a saved model for weight initialization.
        * max_length (optional): Maximum input sequence length.
        * num_class (optional): Number of classes for classification.
        * drop_rate (optional): Dropout rate in the dense layer.
        * epochs (optional): Number of training epochs.
        * batch_size (optional): Batch size for training.
        * learning_rate (optional): Optimizer learning rate.
        * validation_split (optional): Fraction of data for validation.
        * validation_data (optional): Data to use for validation.
        * callbacks (optional): list of tuples with callbacks parameters.
        * parallel_gpu (optional): Flag to use parallel GPU training.
        
    Methods:
        * fit(X, y): Train the model with text (X) and labels (y).
        * predict(X): Predict class labels for the given text (X).
        * predict_proba(X): Predict class probabilities for the given text (X).
        * classification_score(X, y): Compute the weighted F1 score for predictions.
        * cross_validate(X, y, cv=10): Calculate cross-validated scores.
        * save(name): Save the model to the specified path.
        * load(name, parallel_gpu=False): Load a model from the specified path.
        
    Example usage:
        # Initialize the BERT classifier
        bert_classifier = TFbertClassifier(base_name='bert-base-uncased', epochs=3, batch_size=32)
        # Fit the model
        bert_classifier.fit(train_texts, train_labels)
        # Predict labels
        predictions = bert_classifier.predict(test_texts)
        # Save the model
        bert_classifier.save('my_bert_model')
        # Evaluate the model
        f1_score = bert_classifier.classification_score(test_texts, test_labels)
        # Load a model
        bert_classifier.load('my_bert_model')
        
* MLClassifier Class
    A machine learning classifier for text classification using traditional algorithms.

    Constructor Parameters:
        * base_name: Identifier for the base ML model (e.g., 'linearSVC', 'logisticregression').
        * from_trained (optional): Path to a pre-trained model to load.
        * vec_method (optional): Tokenization/Vectorization method ('tfidf', 'skipgram', 'cbow').
        * **kwargs: Additional arguments for the sklearn classifier specified in base_name.

    Methods:
        * fit(X, y): Train the model on text data (X) and labels (y).
        * predict(X): Predict class labels for text data (X).
        * predict_proba(X): Predict class probabilities for text data (X), if supported.
        * classification_score(X, y): Calculate the weighted F1 score for the given predictions.
        * cross_validate(X, y, cv=10): Calculate cross-validated scores.
        * save(name): Save the trained model to a file.
        * load(name): Load a model from a file.
        
    Example usage:
        # Initialize the ML classifier with a logistic regression model
        ml_classifier = MLClassifier(base_name='logisticregression')
        # Train the model
        ml_classifier.fit(train_texts, train_labels)
        # Make predictions
        predictions = ml_classifier.predict(test_texts)
        # Evaluate the model
        f1_score = ml_classifier.classification_score(test_texts, test_labels)
        # Save the model
        ml_classifier.save('my_logistic_regression_model')
        # Load the model
        ml_classifier.save('my_logistic_regression_model')
"""


from transformers import TFAutoModel, AutoTokenizer, CamembertTokenizer, FlaubertTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from src.text.vectorizers import CBowVectorizer, SkipGramVectorizer

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

from joblib import load, dump

import os
import time

import config


def build_bert_model(base_model, from_trained=None, max_length=256, num_class=27, drop_rate=0.0, activation='softmax', strategy=None):
    """
    build_bert_model(base_model, from_trained=None, max_length=256, num_class=27, drop_rate=0.0, activation='softmax', strategy=None)

    Builds a BERT model for classification tasks.

    Arguments:

    * base_model: The base BERT model to be used for creating the classification model.
    * from_trained (optional): name of a pre-trained model to load weights from. Default is None.
    * max_length (int, optional): Maximum length of input sequences. Default is 256.
    * num_class (int, optional): Number of classes for the output layer. Default is 27.
    * drop_rate (float, optional): Dropout rate to be applied to the dense layer. Default is 0.0.
    * activation (str, optional): Activation function for the output layer. Default is 'softmax'.
    * strategy: The TensorFlow distribution strategy to be used for model training. This is necessary for leveraging GPU or multi-GPU setups.

    Returns:
    The constructed Keras Model object.
    """
    with strategy.scope():
        # Input layer and tokenizer layer
        input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(max_length,),
                               dtype='int32', name='attention_mask')

        # Base transformer model
        base_model._name = 'txt_base_layers'
        transformer_layer = base_model(
            {'input_ids': input_ids, 'attention_mask': attention_mask})
        x = transformer_layer[0][:, :, :]
        x = x[:, 0, :]

        # Classification head
        # x = Dense(128, activation='relu', name='txt_Dense_top_1')(x)
        # x = Dropout(rate=drop_rate, name='txt_Drop_out_top_1')(x)
        output = Dense(num_class, activation=activation,
                       name='txt_classification_layer')(x)

        # Construct the final model
        model = Model(inputs={'input_ids': input_ids,
                      'attention_mask': attention_mask}, outputs=output)

        if from_trained is not None:
            trained_model_path = os.path.join(
                config.path_to_models, 'trained_models', from_trained)
            print("loading weights from ", from_trained)
            model.load_weights(trained_model_path + '/weights.h5',
                               by_name=True, skip_mismatch=True)

    return model


class TFbertClassifier(BaseEstimator, ClassifierMixin):
    """
    TFbertClassifier(base_name='camembert-base', from_trained=None, max_length=256,
                        num_class=27, drop_rate=0.2, epochs=1, batch_size=32, 
                        learning_rate=5e-5, callbacks=None, parallel_gpu=False)

    A TensorFlow BERT classifier that wraps around BERT models for text classification tasks, 
    implementing the scikit-learn estimator interface.

    Constructor Arguments:

    * base_name (str, optional): Identifier for the pre-trained base BERT model. Tested model 
      are 'camembert-base', 'camembert/camembert-base-ccnet'. Default is 'camembert-base'.
    * from_trained (optional): Name of a previously saved model in config.path_to_models. Default is None.
    * max_length (int, optional): Maximum sequence length. Default is 256.
    * num_class (int, optional): Number of output classes. Default is 27.
    * drop_rate (float, optional): Dropout rate for the classification head. Default is 0.2.
    * epochs (int, optional): Number of training epochs. Default is 1.
    * batch_size (int, optional): Batch size for training. Default is 32.
    * learning_rate (float, optional): Learning rate for the optimizer. Default is 5e-5.
    * lr_min(float, optional): minimal learning rate if there is a learning rate schedule. Default is None.
    * lr_decay_rate(float, optional): decay rate of the learning rate at every epoch. Default is 1.0.
    * validation_split: fraction of the data to use for validation during training. Default is 0.0.
    * validation_data: a tuple with (features, labels) data to use for validation during training. Default is 0.0.
    * callbacks: A list of tuples with the name of a Keras callback and a dictionnary with matching
      parameters. Example: [('EarlyStopping', {'monitor':'loss', 'min_delta': 0.001, 'patience':2})].
      Default is None.
    * parallel_gpu (bool, optional): Whether to use parallel GPUs. Default is False.

    Methods:

    * fit(X, y): Trains the model on the provided dataset.
    * predict(X): Predicts the class labels for the given input.
    * predict_proba(X): Predicts class probabilities for the given input.
    * classification_score(X, y): Calculates weigthed f1-score for the given input and labels.
    * cross_validate(X, y, cv=10): Calculate cross-validated scores with sklearn cross_validate function
    * save(name): Saves the model to the directory specified in config.path_to_models.
    * load(name, parallel_gpu=False): Loads a model from the directory specified in config.path_to_models.

    Example sage:
    classifier = TFbertClassifier(base_name='bert-base-uncased', epochs=3, batch_size=32)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    f1score = classifier.classification_score(test_data, test_labels)
    classifier.save('my_bert_model')

    """

    def __init__(self, base_name='camembert-base', from_trained=None,
                 max_length=256, num_class=27, drop_rate=0.2,
                 epochs=1, batch_size=32, learning_rate=5e-5, lr_decay_rate=1, lr_min=None,
                 validation_split=0.0, validation_data=None,
                 callbacks=None, parallel_gpu=False):
        """
        Constructor: __init__(self, base_name='camembert-base', from_trained=None, max_length=256, 
                              num_class=27, drop_rate=0.2, epochs=1, batch_size=32, learning_rate=5e-5, 
                              callbacks=None, parallel_gpu=False)

        Initializes a new instance of the TFbertClassifier.

        Arguments:

        * base_name: The identifier for the base BERT model. Tested base model are 'camembert-base',
          'camembert-base-ccnet'. Default is 'camembert-base'.
        * from_trained: Optional path to a directory containing a pre-trained model from which weights will be loaded.
        * max_length: The sequence length that the tokenizer will generate. Default is 256.
        * num_class: The number of classes for the classification task. Default is 27.
        * drop_rate: Dropout rate for the classification head. Default is 0.2.
        * epochs: The number of epochs to train the model. Default is 1.
        * batch_size: Batch size for training. Default is 32.
        * learning_rate: Learning rate for the optimizer. Default is 5e-5.
        * lr_decay_rate: factor by which the learning rate is multiplied at the end of every epoch. Default is 1 (no decay).
        * lr_min: minimal learning rate if there is a learning rate schedule. Default is None (no minimum).
        * validation_split: fraction of the data to use for validation during training. Default is 0.0.
        * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
        * callbacks: A list of tuples with the name of a Keras callback and a dictionnary with matching
          parameters. Example: ('EarlyStopping', {'monitor':'loss', 'min_delta': 0.001, 'patience':2}).
          Default is None.
        * parallel_gpu: Flag to indicate whether to use parallel GPU support. Default is False.

        Returns:

        An instance of TFbertClassifier.
        """
        # defining the parallelization strategy
        if parallel_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        # Defining attributes
        self.max_length = max_length
        self.base_name = base_name
        self.from_trained = from_trained
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_min = lr_min
        self.lr_decay_rate = lr_decay_rate
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.callbacks = callbacks
        self.parallel_gpu = parallel_gpu
        self.history = []

        # Building model and tokenizer
        self.model, self.tokenizer = self._getmodel(from_trained)

        # For sklearn, adding attribute finishing with _ to indicate
        # that the model has already been fitted
        if from_trained is not None:
            self.is_fitted_ = True

    def _getmodel(self, from_trained=None):
        """
        Internal method to load or initialize the model and tokenizer.
        """
        base_model_path = os.path.join(
            config.path_to_models, 'base_models', self.base_name)

        # Loading the pre-trained bert model and its tokenizer
        with self.strategy.scope():
            # If the hugginface pretrained model hasn't been yet saved locally,
            # we load and save it from HuggingFace
            if not os.path.isdir(base_model_path):
                print("loading from Huggingface")

                # Checking if we need to load from pytorch weights
                frompt_list = ['flaubert', 'camembert-base-ccnet']
                if any(name in self.base_name.lower() for name in frompt_list):
                    from_pt = True
                else:
                    from_pt = False

                # Loading model and tokenizer
                if 'camembert' in self.base_name.lower():
                    # We'll load camembert model from there.
                    loadbertfrom = 'almanach/'
                    base_model = TFAutoModel.from_pretrained(
                        loadbertfrom + self.base_name, from_pt=from_pt)
                    base_model.save_pretrained(base_model_path)
                    tokenizer = CamembertTokenizer.from_pretrained(
                        loadbertfrom + self.base_name)
                    tokenizer.save_pretrained(base_model_path)
                elif 'flaubert' in self.base_name.lower():
                    loadbertfrom = 'flaubert/'
                    base_model = TFAutoModel.from_pretrained(
                        loadbertfrom + self.base_name, from_pt=from_pt)
                    base_model.save_pretrained(base_model_path)
                    tokenizer = FlaubertTokenizer.from_pretrained(
                        loadbertfrom + self.base_name)
                    tokenizer.save_pretrained(base_model_path)
            else:
                print("loading from Local")
                if 'camembert' in self.base_name.lower():
                    base_model = TFAutoModel.from_pretrained(base_model_path)
                    tokenizer = CamembertTokenizer.from_pretrained(
                        base_model_path)
                elif 'flaubert' in self.base_name.lower():
                    base_model = TFAutoModel.from_pretrained(base_model_path)
                    tokenizer = FlaubertTokenizer.from_pretrained(
                        'flaubert/flaubert_base_uncased')

        # Building the keras model
        model = build_bert_model(base_model=base_model, from_trained=from_trained,
                                 max_length=self.max_length, num_class=self.num_class,
                                 drop_rate=self.drop_rate, activation='softmax',
                                 strategy=self.strategy)

        return model, tokenizer

    def _lrscheduler(self, epoch):
        """ 
        Internal method for learning rate scheduler
        """
        lr = self.learning_rate * self.lr_decay_rate**epoch

        # the learning is not allowed to be smaller than self.lr_min
        if self.lr_min is not None:
            lr = max(self.lr_min, lr)
        return lr

    def fit(self, X, y):
        """
        Trains the model on the provided dataset.

        Arguments:
        * X: The input text for training. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens"
        * y: The target labels for training.

        Returns:
        The instance of TFbertClassifier after training.
        """

        if self.epochs > 0:
            # Initialize validation data placeholder
            dataset_val = None

            if self.validation_split > 0:
                # Splitting data for validation as necessary
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, random_state=123)
                # Fetching the dataset generator for validation
                dataset_val = self._getdataset(X_val, y_val, training=False)
            elif self.validation_data is not None:
                # If validation data are provided in self.validation_data, we fetch those
                dataset_val = self._getdataset(
                    self.validation_data[0], self.validation_data[1], training=True)
                X_train, y_train = X, y
            else:
                # Use all data for training if validation split is 0
                X_train, y_train = X, y

            # Fetching the training dataset generator
            dataset = self._getdataset(X_train, y_train, training=True)

            with self.strategy.scope():
                # defining the optimizer
                optimizer = Adam(learning_rate=self.learning_rate)

                # Creating callbacks based on self.callback
                callbacks = [tf.keras.callbacks.LearningRateScheduler(
                    schedule=self._lrscheduler)]
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback_api = getattr(tf.keras.callbacks, callback[0])
                        callbacks.append(callback_api(**callback[1]))

                # Compiling
                self.model.compile(
                    optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                # Fitting the model
                fit_args = {'epochs': self.epochs, 'callbacks': callbacks}
                if dataset_val is not None:
                    fit_args['validation_data'] = dataset_val

                start_time = time.time()
                self.history = self.model.fit(dataset, **fit_args)
                self.fit_time = time.time() - start_time
        else:
            # if self.epochs = 0, we just pass the model, considering it has already been trained
            self.history = []

        # For sklearn, adding attribute finishing with _ to indicate
        # that the model has already been fitted
        self.is_fitted_ = True

        # For gridsearchCV and other sklearn method we need a classes_ attribute
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Arguments:
        * X: The text data for prediction. Can be an array like a pandas series, 
          a dataframe with text in column "tokens" or a single string

        Returns:
        An array of predicted class labels.
        """

        if not isinstance(X, str):
            dataset = self._getdataset(X, training=False)
            dataset = dataset.batch(self.batch_size)
        else:
            X_tokenized = self.tokenizer(
                X, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")
            dataset = {"input_ids": X_tokenized['input_ids'],
                       "attention_mask": X_tokenized['attention_mask']}

        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input data.

        Arguments:
        * X: The text data for which to predict class probabilities.
          Can be an array like a pandas series, a dataframe with 
          text in column "tokens"  or a single string

        Returns:
        An array of class probabilities for each input instance.
        """

        if not isinstance(X, str):
            dataset = self._getdataset(X, training=False)
            dataset = dataset.batch(self.batch_size)
        else:
            X_tokenized = self.tokenizer(
                X, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")
            dataset = {"input_ids": X_tokenized['input_ids'],
                       "attention_mask": X_tokenized['attention_mask']}

        probs = self.model.predict(dataset)

        return probs

    def _getdataset(self, X, y=None, training=False):
        """
        Internal method to prepare a TensorFlow dataset from the input data.
        """
        # Fetching text data if X is a dataframe
        if isinstance(X, pd.DataFrame):
            X_txt = X['tokens']
            X_txt = X_txt.tolist()
        else:
            X_txt = X

        # Tokenizing the text with the bert tokenizer
        X_tokenized = self.tokenizer(
            X_txt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf"
        )

        # Dataset from tokenized text, with or without labels depending on whether
        # we use it for training or not
        if training:
            dataset = tf.data.Dataset.from_tensor_slices(
                ({"input_ids": X_tokenized['input_ids'], "attention_mask": X_tokenized['attention_mask']}, y))
            dataset = dataset.shuffle(buffer_size=1000).batch(
                self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                {"input_ids": X_tokenized['input_ids'], "attention_mask": X_tokenized['attention_mask']})

        return dataset

    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y

        Arguments:
        * X: The text data for which to predict class probabilities.
          Can be an array like a pandas series, or a dataframe with 
          text in column "tokens"
        * y: The target labels to predict.

        Returns:
        The average weighted f1-score. Also save scores in classification_results
        and f1score attributes
        """

        # predict class labels for the input text X
        pred = self.predict(X)

        # Save classification report
        self.classification_results = classification_report(
            y, pred, zero_division=0)

        # Build confusion matrix
        self.confusion_mat = confusion_matrix(y, pred, normalize=None)

        # Save weighted f1-score
        self.f1score = f1_score(y, pred, average='weighted', zero_division=0)

        return self.f1score

    def cross_validate(self, X, y, cv=10, n_jobs=None):
        """
        Computes cross-validated scores for the given input X and class labels y

        Arguments:
        * X: The text data for which to cross-validate predictions.
          Can be an array like a pandas series, or a dataframe with 
          text in column "tokens"
        * y: The target labels to predict.
        * cv: Number of folds. Default is 10.
        * n_jobs: number of workers to parallelize on. Default is None.

        Returns:
        The cross-validate scores as returned by sklearn cross_validate 
        function. These scores are saved in the cv_scores attributes
        """
        cvsplitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=123)
        self.cv_scores = cross_validate(
            self, X, y, scoring='f1_weighted', cv=cvsplitter, n_jobs=n_jobs, verbose=0, return_train_score=True)

        return self.cv_scores['test_score'].mean()

    def save(self, name):
        """
        Saves the model to the directory specified in notebook.config file (config.path_to_models).

        Arguments:
        * name: The name to be used for saving the model.
        """
        # path to the directory where the model will be saved
        save_path = os.path.join(config.path_to_models, 'trained_models', name)

        # Creating it if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Saving model's weights to that location
        self.model.save_weights(os.path.join(save_path, 'weights.h5'))

        # Saving the model except for keras objects which are not serialized
        # by joblib
        model_backup = self.model
        tokenizer_backup = self.tokenizer
        history_backup = self.history
        strategy_backup = self.strategy

        self.model = []
        self.tokenizer = []
        self.history = []
        self.strategy = []

        # Saving the model (without keras objects) with joblib
        dump(self, os.path.join(save_path, 'model.joblib'))

        # Restoring the keras objects
        self.model = model_backup
        self.tokenizer = tokenizer_backup
        self.history = history_backup
        self.strategy = strategy_backup

    def load(self, name, parallel_gpu=False):
        """
        Loads a model from the directory specified in notebook.config file (config.path_to_models).

        Arguments:
        * name: The name of the saved model to load.
        * parallel_gpu: Flag to indicate whether to initialize the model 
          for parallel GPU usage.
        """
        # path to the directory where the model to load was saved
        model_path = os.path.join(
            config.path_to_models, 'trained_models', name)

        # Loading the model from there
        loaded_model = load(os.path.join(model_path, 'model.joblib'))

        # tf.distribute.MirroredStrategy is not saved by joblib
        # so we need to update it here
        if parallel_gpu:
            loaded_model.strategy = tf.distribute.MirroredStrategy()
        else:
            loaded_model.strategy = tf.distribute.OneDeviceStrategy(
                device="/gpu:0")

        # Loading the model and its tokenizer
        loaded_model.model, loaded_model.tokenizer = loaded_model._getmodel(
            name)

        return loaded_model


class MLClassifier(BaseEstimator, ClassifierMixin):
    """
    MLClassifier(base_name='linearSVC', from_trained=None, vec_method='tfidf', **kwargs)

    A machine learning classifier that supports various traditional ML algorithms for text classification,
    following the scikit-learn estimator interface.

    Constructor Arguments:

    * base_name (str, optional): Identifier for the base machine learning model. 
        Supported models include 'linearSVC', 'logisticregression', 'multinomialnb', 
        and 'randomforestclassifier'. Default is 'linearSVC'.
    * from_trained (optional): Path to previously saved model. Default is None.
    * vec_method (str, optional): Vectorization method. One of tfidf, skipgram or cbow.
        Default is 'tfidf'.
    * **kwargs: arguments accepted by the chosen sklearn classifier sepcified in
        base_name

    Methods:
    * fit(X, y): Trains the model on the provided dataset.
    * predict(X): Predicts the class labels for the given input.
    * predict_proba(X): Predicts class probabilities for the given input (if predict_proba
      is available for the chosen classifier).
    * classification_score(X, y): Calculates weigthed f1-score for the given input and labels.
    * cross_validate(X, y, cv=10): Calculate cross-validated scores with sklearn cross_validate function
    * save(name): Saves the model to the directory specified in config.path_to_models.

    Example usage:
    ml_classifier = MLClassifier(base_name='logisticregression', vec_method = 'tfidf')
    ml_classifier.fit(train_texts, train_labels)
    predictions = ml_classifier.predict(test_texts)
    f1score = ml_classifier.classification_score(test_texts, test_labels)
    ml_classifier.save('my_ml_model')

    """

    def __init__(self, base_name='linearSVC', from_trained=None, vec_method='tfidf', vec_params=None, **kwargs):
        """
        Constructor: __init__(self, base_name='linearSVC', from_trained=None, vec_method='tfidf', vec_params = None, **kwargs)
        Initializes a new instance of the MLClassifier.

        Arguments:
        * base_name: The identifier for the sklearn machine learning model. Supported models include 
          'linearSVC', 'logisticregression', 'multinomialnb', and 'randomforestclassifier'. Default is 'linearSVC'.
        * from_trained: Optional; name of a model previously saved in config.path_to_models and from which 
          the model state will be loaded.
        * vec_method: Tokenization/Vectorization method. Supported methods are 'tfidf', 'skipgram' and
          'cbow'. Default is 'tfidf'.
        * vec_params: dictionnary of parameters for the vectorization method.
        * **kwargs: Additional keyword arguments that will be passed to the underlying sklearn model.

        Functionality:
        Initializes the classifier based on the specified base_name and configuration, sets up the tokenizer/vectorizer, 
        and prepares the model for training or inference as specified.
        """
        self.base_name = base_name
        self.from_trained = from_trained
        self.vec_method = vec_method
        self.vec_params = vec_params
        self.kwargs = kwargs

        if self.from_trained is not None:
            # loading previously saved model if provided
            self = self.load(self.from_trained)
        else:
            # Initialize the model according to base_name and kwargs
            if base_name.lower() == 'linearsvc':
                self.model = LinearSVC(dual='auto', **kwargs)
            elif base_name.lower() == 'svc':
                self.model = SVC(**kwargs)
            elif base_name.lower() == 'logisticregression':
                self.model = LogisticRegression(dual=False, **kwargs)
            elif base_name.lower() == 'multinomialnb':
                self.model = MultinomialNB(**kwargs)
            elif base_name.lower() == 'randomforestclassifier':
                self.model = RandomForestClassifier(**kwargs)
            elif base_name.lower() == 'xgboost':
                self.model = xgb.XGBClassifier(**kwargs)
            elif base_name.lower() == 'dummyclassifier':
                self.model = DummyClassifier(**kwargs)

            model_params = self.model.get_params()
            for param, value in model_params.items():
                setattr(self, param, value)

            if vec_method.lower() == 'tfidf':
                self.vectorizer = TfidfVectorizer(**(vec_params or {}))
            elif vec_method.lower() == 'cbow':
                self.vectorizer = CBowVectorizer(**(vec_params or {}))
            elif vec_method.lower() == 'skipgram':
                self.vectorizer = SkipGramVectorizer(**(vec_params or {}))

        # Only make predict_proba available if self.model
        # has such method implemented
        if hasattr(self.model, 'predict_proba'):
            self.predict_proba = self._predict_proba

    def get_params(self, deep=True):
        # Return all parameters, including kwargs
        params = super().get_params(deep=deep)
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        # Set parameters including kwargs
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Trains the model on the provided dataset.

        Arguments:
        * X: The input text for training. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens"
        * y: The target labels for training.

        Returns:
        * The instance of MLClassifier after training.
        """

        X_vec = self._getdataset(X, training=True)

        start_time = time.time()
        self.model.fit(X_vec, y)
        self.fit_time = time.time() - start_time

        self.classes_ = np.unique(y)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Arguments:
        * X: The input text for prediction. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens"

        Returns:
        An array of predicted class labels.
        """
        X_vec = self._getdataset(X, training=False)
        pred = self.model.predict(X_vec)
        return pred

    def _predict_proba(self, X):
        """
        Predicts class probabilities for the given text data, if the underlying 
        model supports probability predictions.

        Arguments:
        * X: The input text for prediction. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens"

        Returns:
        An array of class probabilities for each input instance.
        """
        X_vec = self._getdataset(X, training=False)
        probs = self.model.predict_proba(X_vec)

        return probs

    def _getdataset(self, X, y=None, training=False):
        """
        Vectorizes the text data using the chosen vectorizer.

        Arguments:
        * X: The input text. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens"
        * training: boolean. Whether or not thw vectorizer will be fitted to the input text first

        Returns:
        The vectorized input text
        """
        # Fetching text data if X is a dataframe
        if isinstance(X, pd.DataFrame):
            X_txt = X['tokens']
        elif isinstance(X, str):
            X_txt = [X]
        else:
            X_txt = X

        if training:
            X_vec = self.vectorizer.fit_transform(X_txt)
        else:
            X_vec = self.vectorizer.transform(X_txt)

        return X_vec

    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y

        Arguments:
        * X: The text data for which to predict classes.
          Can be an array like a pandas series, or a dataframe with 
          text in column "tokens"
        * y: The target labels to predict.

        Returns:
        The average weighted f1-score. Also save scores in classification_results
        and f1score attributes and confusion matrix in confusion_mat
        """

        # predict class labels for the input text X
        pred = self.predict(X)

        # Save classification report
        self.classification_results = classification_report(
            y, pred, zero_division=0)

        # Build confusion matrix
        self.confusion_mat = confusion_matrix(y, pred, normalize=None)

        # Save weighted f1-score
        self.f1score = f1_score(y, pred, average='weighted', zero_division=0)

        return self.f1score

    def cross_validate(self, X, y, cv=10, n_jobs=None):
        """
        Computes cross-validated scores for the given input X and class labels y

        Arguments:
        * X: The text data for which to cross-validate predictions.
          Can be an array like a pandas series, or a dataframe with 
          text in column "tokens"
        * y: The target labels to predict.
        * cv: Number of folds. Default is 10.
        * n_jobs: number of workers to parallelize on. Default is None.

        Returns:
        The cross-validate scores as returned by sklearn cross_validate 
        function. These scores are saved in the cv_scores attributes
        """
        cvsplitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=123)
        self.cv_scores = cross_validate(
            self, X, y, scoring='f1_weighted', cv=cvsplitter, n_jobs=n_jobs, verbose=0, return_train_score=True)

        return self.cv_scores['test_score'].mean()

    def save(self, name):
        """
        Saves the model to the directory specified in notebook.config file (config.path_to_models).

        Arguments:
        * name: The name to be used for saving the model in config.path_to_models.
        """
        # path to the directory where the model will be saved
        save_path = os.path.join(config.path_to_models, 'trained_models', name)

        # Creating it if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Saving the model to that location
        dump(self, os.path.join(save_path, 'model.joblib'))

    def load(self, name):
        """
        Loads a model from the directory specified in notebook.config file (config.path_to_models).

        Arguments:
        * name: The name of the saved model to load.
        """
        # path to the directory where the model to load was saved
        model_path = os.path.join(
            config.path_to_models, 'trained_models', name)

        # Returning the model from there
        loaded_object = load(os.path.join(model_path, 'model.joblib'))

        return loaded_object
