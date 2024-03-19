"""
Class implementation for constructing and utilizing deep learning models for image classification. 
Below is a summary documentation of the ImgClassifier class and its methods

* ImgClassifier Class
    A wrapper class that allows using various deep learning architectures like ViT, EfficientNet, etc., 
    for image classification within a scikit-learn-like interface.

    Constructor Parameters:
        * base_name: Name of the base architecture.
        * from_trained: Path to a previously saved model.
        * img_size: Tuple representing the input image size.
        * num_class: Number of output classes for classification.
        * drop_rate: Dropout rate for regularization.
        * epochs: Number of epochs to train the model.
        * batch_size: Batch size for training.
        * learning_rate: Learning rate for the optimizer.
        * augmentation_params: Parameters for data augmentation.
        * validation_split: Fraction of data to be used for validation.
        * validation_data: Data to use for validation during training.
        * callbacks: Callbacks to use during training.
        * parallel_gpu: Whether to use parallel GPU support.

    Methods:
        * fit(X, y): Trains the model on the provided dataset.
        * predict(X): Predicts class labels for the input samples.
        * predict_proba(X): Predicts class probabilities for the input samples.
        * classification_score(X, y): Calculates weighted F1-score for the predictions.
        * cross_validate(X, y, cv=10): Calculate cross-validated scores.
        * save(name): Saves the model and its configuration.
        * load(name, parallel_gpu=False): Loads a saved model and its configuration.
        
    Example Usage:
        # Initialize the classifier with VGG16 as the base model
        img_classifier = ImgClassifier(base_name='vgg16', num_class=10, epochs=5, batch_size=32)
        # Fit the model
        img_classifier.fit(train_images, train_labels)
        # Predict on new data
        predictions = img_classifier.predict(test_images)
        # Evaluate the model
        f1score = img_classifier.classification_score(test_images, test_labels)
        # Save the trained model
        img_classifier.save('vgg16_trained_model')
        # Load a model
        img_classifier.load('vgg16_trained_model')
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vit_keras import vit

import numpy as np
import pandas as pd
import cv2

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

from joblib import load, dump

import os
import time

import config


def build_Img_model(base_model, from_trained=None, img_size=(224, 224, 3), num_class=27, drop_rate=0.0, activation='softmax', strategy=None):
    """
    Creates an image classification model with optional pre-trained weights.

    Arguments:
    * base_model: The base model to use for feature extraction (vit, vggg, resnet...).
    * from_trained (optional): Identifier for loading pre-trained model weights.
    * img_size (tuple, optional): The size of the input images. Default is (224, 224, 3).
    * num_class (int, optional): The number of classes for the output layer. Default is 27.
    * drop_rate (float, optional): Dropout rate to be applied before the output layer. Default is 0.0.
    * activation (str, optional): Activation function for the output layer. Default is 'softmax'.
    * strategy: The TensorFlow distribution strategy to be used.

    Returns:
    A tf.keras.Model instance with the constructed image classification model.

    Example usage:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
    model = build_Img_model(base_model, img_size=(224, 224, 3), num_class=10, strategy=strategy)
    """

    with strategy.scope():
        inputs = Input(shape=img_size, name='inputs')
        base_model._name = 'img_base_layers'
        x = base_model(inputs)

        # Adding an average pooling if it's a convNet
        if len(base_model.output_shape) == 4:
            x = GlobalAveragePooling2D()(x)
        elif len(base_model.output_shape) == 2:
            x = x[0][:, :, :]
            x = x[:, 0, :]

        x = Dense(128, activation='relu', name='img_Dense_top_1')(x)
        x = Dropout(rate=drop_rate, name='img_Drop_out_top_1')(x)
        outputs = Dense(num_class, activation=activation,
                        name='img_classification_layer')(x)
        model = Model(inputs=inputs, outputs=outputs)

        if from_trained is not None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            trained_model_path = os.path.join(
                config.path_to_models, 'trained_models', from_trained)
            print("loading weights from ", from_trained)
            model.load_weights(trained_model_path + '/weights.h5',
                               by_name=True, skip_mismatch=True)

    return model


class ImgClassifier(BaseEstimator, ClassifierMixin):
    """
     Image classification model built upon TensorFlow, capable of incorporating various pre-trained architectures like 
     Vision Transformer (ViT), EfficientNet, ResNet, VGG16, and VGG19. This class is designed to seamlessly integrate 
     with scikit-learn's estimator interface, supporting custom image sizes, classification tasks with multiple classes, 
     dropout regularization, and TensorFlow's strategies for distributed training.

    Constructor Arguments:
    * base_name: The name of the base model architecture. Default is 'vit_b16'.
    * from_trained: Name of a previously saved model in config.path_to_models. Default is None.
    * img_size: Tuple representing the size of the input images. Default is (224, 224, 3).
    * num_class: Number of classes for classification. Default is 27.
    * drop_rate: Dropout rate for regularization. Default is 0.2.
    * epochs: Number of training epochs. Default is 1.
    * batch_size: Batch size for training. Default is 32.
    * learning_rate: Learning rate for the optimizer. Default is 5e-5.
    * lr_decay_rate: decay rate of the learning rate at every epoch.
    * lr_min: minimal learning rate if there is a learning rate schedule. Default is None (no minimum).
    * augmentation_params: Dictionary specifying parameters for data augmentation. Default is None, which applies a standard set of augmentations.
    * validation_split: fraction of the data to use for validation during training. Default is 0.0.
    * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
    * callbacks: A list of tuples with the name of a Keras callback and a dictionnary with matching
      parameters. Example: ('EarlyStopping', {'monitor':'loss', 'min_delta': 0.001, 'patience':2}).
      Default is None.
    * parallel_gpu: Whether to use TensorFlow's parallel GPU training capabilities. Default is True.

    Methods:
    * fit(self, X, y): Trains the model on a dataset.
    * predict(self, X): Predicts class labels for the input samples.
    * predict_proba(self, X): Predicts class probabilities for the input samples.
    * classification_score(X, y): Calculates weigthed f1-score for the given input and labels.
    * cross_validate(X, y, cv=10): Calculate cross-validated scores with sklearn cross_validate function.
    * save(self, name): Saves the model weights and configuration.
    * load(self, name, parallel_gpu=False): Loads a previously saved model configuration and weights.

    Example usage:
    classifier = ImgClassifier(base_name='vgg16', num_class=27, epochs=10, batch_size=32, parallel_gpu=True)
    classifier.fit(train_images, train_labels)
    predictions = classifier.predict(test_images)
    classification_score(test_images, test_labels)
    classifier.save('vgg16_trained_model')
    """

    def __init__(self, base_name='vit_b16', from_trained=None,
                 img_size=(224, 224, 3), num_class=27, drop_rate=0.2,
                 epochs=1, batch_size=32, learning_rate=5e-5, lr_decay_rate=1, lr_min=None,
                 validation_split=0.0, validation_data=None,
                 augmentation_params=None, callbacks=None, parallel_gpu=True):
        """
        Constructor: __init__(self, base_name='vit_b16', from_trained=None, img_size=(224, 224, 3), num_class=27, drop_rate=0.2, 
                              epochs=1, batch_size=32, learning_rate=5e-5, augmentation_params=None, callbacks=None,
                              parallel_gpu=True)
        Initializes a new instance of the ImgClassifier.

        Arguments:
        * base_name: Identifier for the base model architecture (e.g., 'vit_b16', 'vgg16', 'resnet50').
        * from_trained: Optional Name of a previously saved model in config.path_to_models. Default is None.
        * img_size: The size of input images.
        * num_class: The number of classes for classification.
        * drop_rate: Dropout rate before the final classification layer.
        * epochs: Number of epochs to train for.
        * batch_size: Size of batches for training.
        * learning_rate: Learning rate for the optimizer.
        * lr_decay_rate: factor by which the learning rate is multiplied at the end of every epoch. Default is 1 (no decay).
        * lr_min: minimal learning rate if there is a learning rate schedule. Default is None (no minimum).
        * augmentation_params: a dictionnary with parameters for data augmentation (see ImageDataGenerator).
        * validation_split: fraction of the data to use for validation during training. Default is 0.0.
        * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
        * callbacks: A list of tuples with the name of a Keras callback and a dictionnary with matching
        * parameters. Example: ('EarlyStopping', {'monitor':'loss', 'min_delta': 0.001, 'patience':2}).
          Default is None.
        * parallel_gpu: Whether to use TensorFlow's mirrored strategy for parallel GPU training.
        """
        # defining the parallelization strategy
        if parallel_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        # Defining attributes
        self.img_size = img_size
        self.base_name = base_name
        self.from_trained = from_trained
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_min = lr_min
        self.lr_decay_rate = lr_decay_rate
        if augmentation_params is None:
            augmentation_params = dict(rotation_range=20, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True,
                                       fill_mode='constant', cval=255)
        self.augmentation_params = augmentation_params
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.callbacks = callbacks
        self.parallel_gpu = parallel_gpu
        self.history = []

        # Building model and tokenizer
        self.model, self.preprocessing_function = self._getmodel(from_trained)

        # For sklearn, adding attribute finishing with _ to indicate
        # that the model has already been fitted
        if from_trained is not None:
            self.is_fitted_ = True

    def _getmodel(self, from_trained=None):
        """
        Internal method to initialize or load the base model and set up preprocessing function.
        """
        with self.strategy.scope():
            if 'vit' in self.base_name.lower():
                def default_action(): return print(
                    "base_name should be one of: b16, b32, L16 or L32")
                vit_model = getattr(vit, 'vit_' + self.base_name[-3:], default_action)(image_size=self.img_size[0:2], pretrained=True,
                                                                                       include_top=False, pretrained_top=False)
                base_model = Model(inputs=vit_model.input,
                                   outputs=vit_model.layers[-3].output)
                preprocessing_function = lambda x: (x / 255.0 - np.mean(x / 255.0, keepdims=True)) / np.std(x / 255.0, keepdims=True)
                model_class = None
            elif 'efficientnet' in self.base_name.lower():
                model_class = getattr(
                    tf.keras.applications.efficientnet, self.base_name)
                preprocessing_function = tf.keras.applications.efficientnet.preprocess_input
            elif 'resnet' in self.base_name.lower():
                model_class = getattr(
                    tf.keras.applications.resnet, self.base_name)
                preprocessing_function = tf.keras.applications.resnet.preprocess_input
            elif 'vgg16' in self.base_name.lower():
                model_class = getattr(
                    tf.keras.applications.vgg16, self.base_name.upper())
                preprocessing_function = tf.keras.applications.vgg16.preprocess_input
            elif 'vgg19' in self.base_name.lower():
                model_class = getattr(
                    tf.keras.applications.vgg19, self.base_name.upper())
                preprocessing_function = tf.keras.applications.vgg19.preprocess_input

            if model_class is not None:
                base_model = model_class(
                    weights='imagenet', include_top=False, input_shape=self.img_size)

        model = build_Img_model(base_model=base_model, from_trained=from_trained,
                                img_size=self.img_size, num_class=self.num_class,
                                drop_rate=self.drop_rate, activation='softmax',
                                strategy=self.strategy)

        return model, preprocessing_function

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

        Parameters:
        * X: The image data for training. Can be an array like a pandas series, 
          or a dataframe with column "img_path" containing the full path to the images
        * y: The target labels for training.

        Returns:
        The instance of ImgClassifier after training.
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
        * X: The image data for prediction. Can be an array like a pandas series, 
          or a dataframe with column "img_path" containing the full path to the images

        Returns:
        An array of predicted class labels.
        """
        #checking if X is an image
        if not isinstance(X, np.ndarray):
            dataset = self._getdataset(X, training=False)
        else:
            X = cv2.resize(X, self.img_size[:2])
            X = self.preprocessing_function(X)
            dataset = X.reshape((1,) + X.shape)
            
        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input data.

        Arguments:
        * X: The image data for prediction. Can be an array like a pandas series, 
          a dataframe with column "img_path" containing the full path to the images
          or a single image provided as a numpy array

        Returns:
        An array of class probabilities for each input instance.
        """
        
        #checking if X is an image
        if not isinstance(X, np.ndarray):
            dataset = self._getdataset(X, training=False)
        else:
            X = cv2.resize(X, self.img_size[:2])
            X = self.preprocessing_function(X)
            dataset = X.reshape((1,) + X.shape)
            
        probs = self.model.predict(dataset)
        return probs

    def _getdataset(self, X, y=None, training=False):
        """
        Internal method to prepare a TensorFlow dataset from the input data.
        """
        # Fetching data if X is a dataframe
        if isinstance(X, pd.DataFrame):
            X_img = X['img_path']
        else:
            X_img = X

        if y is None:
            y = 0

        df = pd.DataFrame({'labels': y, 'img_path': X_img})

        if training:
            shuffle = True
            params = self.augmentation_params
        else:
            shuffle = False
            params = dict(rotation_range=0, width_shift_range=0,
                          height_shift_range=0, horizontal_flip=False,
                          fill_mode='constant', cval=255)

        # Data generator for the train and test sets
        # if 'vit' in self.base_name.lower():
        #     data_generator = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True,
        #                                         rotation_range=params['rotation_range'],
        #                                         width_shift_range=params['width_shift_range'],
        #                                         height_shift_range=params['height_shift_range'],
        #                                         horizontal_flip=params['horizontal_flip'],
        #                                         fill_mode=params['fill_mode'],
        #                                         cval=params['cval'])
        # else:
        data_generator = ImageDataGenerator(preprocessing_function=self.preprocessing_function,
                                            rotation_range=params['rotation_range'],
                                            width_shift_range=params['width_shift_range'],
                                            height_shift_range=params['height_shift_range'],
                                            horizontal_flip=params['horizontal_flip'],
                                            fill_mode=params['fill_mode'],
                                            cval=params['cval'])

        dataset = data_generator.flow_from_dataframe(dataframe=df, x_col='img_path', y_col='labels',
                                                     class_mode='raw', target_size=self.img_size[:2],
                                                     batch_size=self.batch_size, shuffle=shuffle)

        return dataset

    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y

        Arguments:
        * X: The image data for which to predict class probabilities.
          Can be an array like a pandas series, or a dataframe with 
          with column "img_path" containing the full path to the images
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
        * X: The image data for which to cross-validate predictions.
          Can be an array like a pandas series, or a dataframe with 
          image path in column "img_path"
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
        preprocessing_function_backup = self.preprocessing_function
        history_backup = self.history
        strategy_backup = self.strategy

        self.model = []
        self.preprocessing_function = []
        self.history = []
        self.strategy = []

        dump(self, os.path.join(save_path, 'model.joblib'))

        self.model = model_backup
        self.preprocessing_function = preprocessing_function_backup
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

        # Re-building the model and loading the weights which has been saved
        # in model_path
        loaded_model.model, loaded_model.preprocessing_function = loaded_model._getmodel(
            name)

        return loaded_model
