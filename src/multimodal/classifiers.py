""" 
Class implementations for building and utilizing multimodal and ensemble models, adeptly combining text and 
image data for comprehensive classification tasks. This documentation provides a summary of the 
TFmultiClassifier and MetaClassifier classes with their main paremeters and methods

* TFmultiClassifier Class
    A TensorFlow-based classifier for multimodal (text and image) data.

    Constructor Parameters:
        * txt_base_name: Identifier for the base text model.
        * img_base_name: Identifier for the base image model.
        * from_trained: Optional, specifies pre-trained model weights.
        * max_length: Maximum sequence length for text inputs.
        * img_size: The size of the input images.
        * augmentation_params: Data augmentation parameters.
        * validation_split: Fraction of data to be used for validation.
        * validation_data: Data to use for validation during training.
        * num_class: The number of output classes.
        * drop_rate: Dropout rate for regularization.
        * epochs: Number of epochs to train the model.
        * batch_size: Batch size for training.
        * learning_rate: Learning rate for the optimizer.
        * validation_split: Fraction of data used for validation.
        * callbacks: Callbacks for training.
        * parallel_gpu: Whether to use parallel GPU support.

    Methods:
        * fit(X, y): Trains the model.
        * predict(X): Predicts class labels for input data.
        * predict_proba(X): Predicts class probabilities.
        * classification_score(X, y): Calculates classification metrics.
        * cross_validate(X, y, cv=10): Calculate cross-validated scores.
        * save(name): Saves the model.
        * load(name, parallel_gpu): Loads a saved model.

    Example Usage:
        X = pd.DataFrame({'text': txt_data, 'img_path': img_data})
        y = labels
        classifier = TFmultiClassifier(txt_base_name='bert-base-uncased', img_base_name='vit_b16', epochs=5, batch_size=2)
        classifier.fit(X, y)
        f1score = classifier.classification_score(X_test, y_test)
        classifier.save('multimodal_model')
        
* MetaClassifier Class
    A wrapper class for various ensemble methods, enabling the combination of multiple classifier models for improved prediction accuracy.

    Constructor Parameters:
        * base_estimators: List of tuples with base estimators and their names.
        * meta_method: The ensemble method to use, such as 'voting', 'stacking', 'bagging', or 'boosting'.
        * from_trained: Optional; path to a previously saved ensemble model.
        * **kwargs: Additional arguments specific to the chosen ensemble meta_method.
    
    Methods:
        * fit(X, y): Trains the ensemble model on the given dataset.
        * predict(X): Predicts class labels for the input data.
        * predict_proba(X): Predicts class probabilities for the input data (if supported by the base models).
        * classification_score(X, y): Calculates the weighted F1-score for the predictions.
        * cross_validate(X, y, cv=10): Calculate cross-validated scores.
        * save(name): Saves the ensemble model and its base models.
        * load(name): Loads the ensemble model and its base models.
        
    Example Usage:
        base_estimators = [('clf1', TFbertClassifier()), ('clf2', MLClassifier())]
        meta_classifier = MetaClassifier(base_estimators=base_estimators, meta_method='voting')
        meta_classifier.fit(X, y)
        predictions = meta_classifier.predict(X_test)
        f1score = classifier.classification_score(X_test, y_test)
        meta_classifier.save('my_ensemble_model')
        meta_classifier.load('my_ensemble_model')
"""


from transformers import TFAutoModel, AutoTokenizer, CamembertTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, Concatenate, BatchNormalization, LayerNormalization, MultiHeadAttention, Add, Flatten
from tensorflow.keras.models import Model, Sequential
from keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

from vit_keras import vit

import numpy as np
import pandas as pd
import cv2

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from joblib import load, dump
import os
import time

import config


def build_multi_model(txt_base_model, img_base_model, from_trained=None, max_length=256, img_size=(224, 224, 3),
                      num_class=27, drop_rate=0.0, activation='softmax', attention_numheads=8,
                      transfo_numblocks=1, strategy=None):
    """
    Creates a multimodal classification model that combines text and image data for prediction tasks.

    Arguments:
    * txt_base_model: Pre-initialized text model (such as BERT) to be used for text feature extraction.
    * img_base_model: Pre-initialized image model (such as Vision Transformer, ViT) for image feature extraction.
    * from_trained (optional): Path or dictionary specifying the name of pre-trained models for text and/or image models. 
      If a dictionary, keys should be 'text' and 'image' with names of the models as values. Name of a pret-trained 
      full model should be passed as a simple string
    * max_length (int, optional): Maximum sequence length for text inputs. Default is 256.
    * img_size (tuple, optional): Size of the input images. Default is (224, 224, 3).
    * num_class (int, optional): Number of classes for the classification task. Default is 27.
    * drop_rate (float, optional): Dropout rate applied in the final layers of the model. Default is 0.0.
    * activation (str, optional): Activation function for the output layer. Default is 'softmax'.
    * attention_numheads (int, optional): number of heads in multi-attention layers. Default is 8.
    * transfo_numblocks (int, optional): number of transformer blocks after fusion. Default is 1.
    * strategy: TensorFlow distribution strategy to be used during model construction.

    Returns:
    A TensorFlow Model instance representing the constructed multimodal classification model.

    Example usage:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    txt_model = TFAutoModel.from_pretrained('bert-base-uncased')
    img_model = vit.vit_b16(image_size=(224, 224), pretrained=True, include_top=False, pretrained_top=False)
    multimodal_model = build_multi_model(txt_base_model=txt_model, img_base_model=img_model, max_length=256, img_size=(224, 224, 3), num_class=10, strategy=strategy)
    """
    with strategy.scope():
        # Bert branch
        input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(max_length,),
                               dtype='int32', name='attention_mask')

        # Bert transformer model
        txt_base_model._name = 'txt_base_layers'
        txt_transformer_layer = txt_base_model(
            {'input_ids': input_ids, 'attention_mask': attention_mask})
        x = txt_transformer_layer[0][:, :, :]
        x = x[:, 0, :]
        x = LayerNormalization(epsilon=1e-6, name='txt_normalization')(x)

        outputs = x
        txt_model = Model(inputs={'input_ids': input_ids,
                          'attention_mask': attention_mask}, outputs=outputs)

        # Loading pre-saved weights to the Bert model if provided
        if from_trained is not None:
            if isinstance(from_trained, dict):
                if 'text' in from_trained.keys():
                    txt_model_path = os.path.join(
                        config.path_to_models, 'trained_models', from_trained['text'])
                    print("loading weights for BERT from ",
                          from_trained['text'])
                    txt_model.load_weights(
                        txt_model_path + '/weights.h5', by_name=True, skip_mismatch=True)

        # ViT transformer model
        input_img = Input(shape=img_size, name='inputs')
        img_base_model._name = 'img_base_layers'
        img_transformer_layer = img_base_model(input_img)
        x = img_transformer_layer[0][:, :, :]
        x = x[:, 0, :]
        x = LayerNormalization(epsilon=1e-6, name='img_normalization')(x)

        outputs = x
        img_model = Model(inputs=input_img, outputs=outputs)

        # Loading pre-saved weights to the Image model if provided
        if from_trained is not None:
            if isinstance(from_trained, dict):
                if 'image' in from_trained.keys():
                    img_model_path = os.path.join(
                        config.path_to_models, 'trained_models', from_trained['image'])
                    print("loading weights for ViT from ",
                          from_trained['image'])
                    img_model.load_weights(
                        img_model_path + '/weights.h5', by_name=True, skip_mismatch=True)

        # Concatenate text and image models
        if transfo_numblocks > 0 and attention_numheads > 0:
            # Output of text and image models before slicing
            img_output = LayerNormalization(
                epsilon=1e-6, name='img_normalization')(img_model.layers[-3].output)
            txt_output = LayerNormalization(
                epsilon=1e-6, name='txt_normalization')(txt_model.layers[-3].output)

            x = txt_output
            embed_dim = x.shape[-1]
            transformer_block = TransformerBlock(
                num_heads=attention_numheads, embed_dim=embed_dim, name='cross-modal_layer')
            x = transformer_block(
                x=txt_output, key=img_output, value=img_output)

            # Adding transformer blocks
            for k in range(transfo_numblocks):
                transformer_block = TransformerBlock(
                    num_heads=attention_numheads, embed_dim=embed_dim, name='transformer_block_' + str(k))
                x = transformer_block(x=x, key=x, value=x)

            # Keeping the first token only
            x = x[:, 0, :]
        else:
            # If no transformer blocks to add, we simply concatenate
            # along the embedding dimension
            x = Concatenate()([txt_model.output, img_model.output])

        # Dense layers for classification
        # x = Flatten()(x)
        x = Dropout(rate=drop_rate, name='multi_Drop_out_top_1')(x)
        x = Dense(units=128, activation='relu', name='Dense_multi_1')(x)
        outputs = Dense(units=num_class, activation=activation,
                        name='multi_classification_layer')(x)

        model = Model(
            inputs=[txt_model.input, img_model.input], outputs=outputs)

        # Loading pre-saved weights to the full model if provided
        if from_trained is not None:
            if not isinstance(from_trained, dict):
                model_path = os.path.join(
                    config.path_to_models, 'trained_models', from_trained)
                print("loading weights for multimodal model from ", from_trained)
                model.load_weights(model_path + '/weights.h5',
                                   by_name=True, skip_mismatch=True)

    return model


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim=None, drop_rate=0.0, name=None):
        super(TransformerBlock, self).__init__(name=name)
        # Default value for ff_dim (dim of intermediate dense layer)
        # is twice the size of the embedding
        if ff_dim is None:
            ff_dim = 2 * embed_dim

        # Create layers:
        # MultiHead attention
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        # Feed-forward with 2 dense
        self.ffn = Sequential([Dense(ff_dim, activation="relu"),
                               Dense(embed_dim)])

        # Normalization and drop-out
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate=drop_rate)
        self.dropout2 = Dropout(rate=drop_rate)

    def call(self, x, key, value, training):
        # Cross-attention
        attn_output, attn_scores = self.att(
            query=x, key=key, value=value, return_attention_scores=True)

        # Caching the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        # Summing input and output of attention head
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = x + attn_output
        out1 = self.layernorm1(attn_output)

        # Passing the output of the Feed-forward module to the
        # feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = ffn_output + out1
        return self.layernorm2(ffn_output)


class MultimodalDataGenerator(Sequence):
    """
    A custom data generator for batching and preprocessing multimodal data (text and images) for training or prediction with a Keras model.

    Constructor Arguments:
    * img_data_generator: An instance of Data genrator for real-time data augmentation and preprocessing of image and text data.
    * img_path: List or Pandas Series containing paths to the images.
    * text_tokenized: A dictionary containing tokenized text data with keys 'input_ids' and 'attention_mask'.
    * labels: Numpy array or Pandas Series containing target labels for the dataset.
    * batch_size (int, optional): Number of samples per batch. Default is 32.
    * target_size (tuple, optional): The dimensions to which all images found will be resized. Default is (224, 224).
    * shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Default is True.

    Methods:
    * __len__: Returns the number of batches per epoch.
    * __getitem__: Returns a batch of data (text and images) and corresponding labels.
    * on_epoch_end: Updates indexes after each epoch if shuffle is True.
    """

    def __init__(self, img_data_generator, img_path, text_tokenized, labels, batch_size=32, target_size=(224, 224), shuffle=True):
        self.img_data_generator = img_data_generator
        self.dataframe = pd.DataFrame(
            {'filename': img_path})  # dataframe.copy()
        self.text_tokenized = text_tokenized
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:min(
            (index + 1) * self.batch_size, len(self.dataframe))]
        batch_indexes_tensor = tf.convert_to_tensor(
            batch_indexes, dtype=tf.int32)

        batch_df = self.dataframe.iloc[batch_indexes]

        img_generator = self.img_data_generator.flow_from_dataframe(dataframe=batch_df, target_size=self.target_size,
                                                                    x_col="filename", y_col=None, class_mode=None,
                                                                    batch_size=len(batch_df), shuffle=False)

        images = np.concatenate([img_generator.next()
                                for _ in range(len(img_generator))], axis=0)

        token_ids = tf.gather(
            self.text_tokenized['input_ids'], batch_indexes_tensor, axis=0)
        attention_mask = tf.gather(
            self.text_tokenized['attention_mask'], batch_indexes_tensor, axis=0)

        labels = self.labels[batch_indexes].values

        return [{"input_ids": token_ids, "attention_mask": attention_mask}, images], labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class TFmultiClassifier(BaseEstimator, ClassifierMixin):
    """
    A TensorFlow-based classifier for multimodal (text and image) data, implementing the scikit-learn estimator interface.

    Constructor Arguments:
    * txt_base_name: Identifier for the base text model (e.g., 'bert-base-uncased').
    * img_base_name: Identifier for the base image model (e.g., 'vit_b16').
    * from_trained: Path or dictionary specifying the name of pre-trained models for text and/or image models. 
      If a dictionary, keys should be 'text' and 'image' with names of the models as values. Name of a pret-trained 
      full model should be passed as a simple string
    * max_length (int, optional): Maximum sequence length for text inputs.
    * img_size (tuple, optional): Size of the input images.
    * augmentation_params: Parameters for data augmentation.
    * validation_split: fraction of the data to use for validation during training. Default is 0.0.
    * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
    * num_class (int, optional): Number of classes for the classification task.
    * attention_numheads (int, optional): number of heads in multi-attention layers. Default is 8.
    * transfo_numblocks (int, optional): number of transformer blocks after fusion. Default is 1.
    * drop_rate (float, optional): Dropout rate applied in the final layers of the model.
    * epochs (int, optional): Number of epochs for training.
    * batch_size (int, optional): Number of samples per batch.
    * learning_rate (float, optional): Learning rate for the optimizer.
    * lr_min: minimal learning rate if there is a learning rate schedule. Default is None (no minimum).
    * lr_decay_rate: decay rate of the learning rate at every epoch.
    * callbacks: List of Keras callbacks to be used during training.
    * parallel_gpu (bool, optional): Whether to use TensorFlow's parallel GPU training capabilities.

    Methods:
    * fit: Trains the multimodal model on a dataset.
    * predict: Predicts class labels for the given input data.
    * predict_proba: Predicts class probabilities for the given input data.
    * classification_score: Computes classification metrics for the given input data and true labels.
    * cross_validate(X, y, cv=10): Calculate cross-validated scores with sklearn cross_validate function.
    * save: Saves the model's weights and tokenizer to the specified directory.
    * load: Loads the model's weights and tokenizer from the specified directory.

    Example Usage:
    X = pd.DataFrame({'text': txt_data, 'img_path': img_data})
    y = labels
    classifier = TFmultiClassifier(txt_base_name='bert-base-uncased', img_base_name='vit_b16', epochs=5, batch_size=2)
    classifier.fit(X, y)
    score = classifier.classification_score(X_test, y_test)
    classifier.save('multimodal_model')
    """

    def __init__(self, txt_base_name='camembert-base', img_base_name='vit_b16', from_trained=None,
                 max_length=256, img_size=(224, 224, 3), augmentation_params=None,
                 num_class=27, drop_rate=0.2, epochs=1, batch_size=32,
                 transfo_numblocks=1, attention_numheads=8,
                 validation_split=0.0, validation_data=None,
                 learning_rate=5e-5, lr_decay_rate=1, lr_min=None,
                 callbacks=None, parallel_gpu=True):
        """
        Constructor: __init__(self, txt_base_name='camembert-base', img_base_name='b16', from_trained = None, 
                              max_length=256, img_size=(224, 224, 3), augmentation_params=None, num_class=27, drop_rate=0.2,
                              epochs=1, batch_size=32, learning_rate=5e-5, callbacks=None, parallel_gpu=True)

        Initializes a new instance of the TFmultiClassifier.

        Arguments:

        * txt_base_name: The identifier for the base BERT model. Tested base model are 'camembert-base',
          'camembert/camembert-base-ccnet'. Default is 'camembert-base'.
        * img_base_name: The identifier for the base vision model architecture (e.g., 'vit_b16', 'vgg16', 'resnet50').
        * from_trained: Path or dictionary specifying the name of pre-trained models for text and/or image models. 
          If a dictionary, keys should be 'text' and 'image' with names of the models as values. Name of a pret-trained 
          full model should be passed as a simple string
        * max_length: The sequence length that the tokenizer will generate. Default is 256.
        * img_size: The size of input images.
        * num_class: The number of classes for the classification task. Default is 27.
        * augmentation_params: a dictionnary with parameters for data augmentation (see ImageDataGenerator).
        * attention_numheads (int, optional): number of heads in multi-attention layers. Default is 8.
        * transfo_numblocks (int, optional): number of transformer blocks after fusion. Default is 1.
        * drop_rate: Dropout rate for the classification head. Default is 0.2.
        * epochs: The number of epochs to train the model. Default is 1.
        * batch_size: Batch size for training. Default is 32.
        * learning_rate: Learning rate for the optimizer. Default is 5e-5.
        * lr_min: minimal learning rate if there is a learning rate schedule. Default is None (no minimum).
        * lr_decay_rate: factor by which the learning rate is multiplied at the end of every epoch. Default is 1 (no decay).
        * validation_split: fraction of the data to use for validation during training. Default is 0.0.
        * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
        * callbacks: A list of tuples with the name of a Keras callback and a dictionnary with matching
          parameters. Example: ('EarlyStopping', {'monitor':'loss', 'min_delta': 0.001, 'patience':2}).
          Default is None.
        * parallel_gpu: Flag to indicate whether to use parallel GPU support. Default is False.

        Returns:

        An instance of TFmultiClassifier.
        """

        # defining the parallelization strategy
        if parallel_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        # Defining attributes
        self.max_length = max_length
        self.img_size = img_size
        self.txt_base_name = txt_base_name
        self.img_base_name = img_base_name
        self.from_trained = from_trained
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.attention_numheads = attention_numheads
        self.transfo_numblocks = transfo_numblocks
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

        # Building model and tokenizer
        self.model, self.tokenizer, self.preprocessing_function = self._getmodel(
            from_trained)

        # For sklearn, adding attribute finishing with _ to indicate
        # that the model has already been fitted
        if from_trained is not None:
            self.is_fitted_ = True

    def _getmodel(self, from_trained=None):
        """
        Internal method to initialize or load the base model and set up preprocessing.
        """
        # path to locally saved huggingface Bert model
        txt_base_model_path = os.path.join(
            config.path_to_models, 'base_models', self.txt_base_name)

        with self.strategy.scope():
            # Loading bert model base
            if not os.path.isdir(txt_base_model_path):
                # If the hugginface pretrained Bert model hasn't been yet saved locally,
                # we load and save it from HuggingFace
                txt_base_model = TFAutoModel.from_pretrained(
                    self.txt_base_name)
                txt_base_model.save_pretrained(txt_base_model_path)
                tokenizer = CamembertTokenizer.from_pretrained(
                    self.txt_base_name)
                tokenizer.save_pretrained(txt_base_model_path)
            else:
                txt_base_model = TFAutoModel.from_pretrained(
                    txt_base_model_path)
                tokenizer = CamembertTokenizer.from_pretrained(
                    txt_base_model_path)

            # Loading ViT model base
            def default_action(): return print(
                "img_base_name should be one of: b16, b32, L16 or L32")
            vit_model = getattr(vit, 'vit_' + self.img_base_name[-3:], default_action)(image_size=self.img_size[0:2], pretrained=True,
                                                                                       include_top=False, pretrained_top=False)
            img_base_model = Model(
                inputs=vit_model.input, outputs=vit_model.layers[-3].output)
            preprocessing_function = lambda x: (x / 255.0 - np.mean(x / 255.0, keepdims=True)) / np.std(x / 255.0, keepdims=True)

        model = build_multi_model(txt_base_model=txt_base_model, img_base_model=img_base_model,
                                  from_trained=from_trained, max_length=self.max_length, img_size=self.img_size,
                                  num_class=self.num_class, drop_rate=self.drop_rate, activation='softmax',
                                  attention_numheads=self.attention_numheads, transfo_numblocks=self.transfo_numblocks,
                                  strategy=self.strategy)

        return model, tokenizer, preprocessing_function

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
        * X: The image and text data for training. Should be a dataframe with columns
          "tokens" containing the text and column "img_path" containing the full paths 
          to the images
        * y: The target labels for training.

        Returns:
        The instance of TFmultiClassifier after training.
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
        * X: The image and text data for prediction. For batches, this should be 
          a dataframe with columns "tokens" containing the text and column 
          "img_path" containing the full paths to the images. For single prediction,
          it should be a dictionnary with keys "text" and "image" containing the text
          as a string and the image as a numpy array

        Returns:
        An array of predicted class labels.
        """
        
        #if X is not a dictionnary, it should be a dataframe
        #if it is a dictionnary, it should provide 'text' as a string
        # and 'image' as a numpy array
        if not isinstance(X, dict):
            dataset = self._getdataset(X, training=False)
        else:
            X_tokenized = self.tokenizer(X['text'], 
                                         padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")
            X_image = cv2.resize(X['image'], self.img_size[:2])
            X_image = self.preprocessing_function(X_image)
            X_image = X_image.reshape((1,) + X_image.shape)
            image_processed = self.preprocessing_function(X_image)
            dataset = [{"input_ids": X_tokenized['input_ids'], "attention_mask": X_tokenized['attention_mask']}, image_processed]
            
        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input data.

        Arguments:
        * X: The image and text data for prediction. For batches, this should be 
          a dataframe with columns "tokens" containing the text and column 
          "img_path" containing the full paths to the images. For single prediction,
          it should be a dictionnary with keys "text" and "image" containing the text
          as a string and the image as a numpy array

        Returns:
        An array of class probabilities for each input instance.
        """
        
        #if X is not a dictionnary, it should be a dataframe
        #if it is a dictionnary, it should provide 'text' as a string
        # and 'image' as a numpy array
        if not isinstance(X, dict):
            dataset = self._getdataset(X, training=False)
        else:
            X_tokenized = self.tokenizer(X['text'], 
                                         padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")
            X_image = cv2.resize(X['image'], self.img_size[:2])
            X_image = self.preprocessing_function(X_image)
            X_image = X_image.reshape((1,) + X_image.shape)
            image_processed = self.preprocessing_function(X_image)
            dataset = [{"input_ids": X_tokenized['input_ids'], "attention_mask": X_tokenized['attention_mask']}, image_processed]
            
        probs = self.model.predict(dataset)

        return probs

    def _getdataset(self, X, y=None, training=False):
        """
        Internal method to prepare a TensorFlow dataset from the input data.
        """
        if y is None:
            y = 0

        df = pd.DataFrame(
            {'labels': y, 'tokens': X['tokens'], 'img_path': X['img_path']})

        if training:
            shuffle = True
            params = self.augmentation_params
        else:
            shuffle = False
            params = dict(rotation_range=0, width_shift_range=0,
                          height_shift_range=0, horizontal_flip=False,
                          fill_mode='constant', cval=255)

        # Data generator for the train and test sets
        img_generator = ImageDataGenerator(preprocessing_function=self.preprocessing_function,
                                           rotation_range=params['rotation_range'],
                                           width_shift_range=params['width_shift_range'],
                                           height_shift_range=params['height_shift_range'],
                                           horizontal_flip=params['horizontal_flip'],
                                           fill_mode=params['fill_mode'],
                                           cval=params['cval'])

        X_tokenized = self.tokenizer(df['tokens'].tolist(
        ), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")

        dataset = MultimodalDataGenerator(img_generator, df['img_path'], X_tokenized, df['labels'],
                                          batch_size=self.batch_size, target_size=self.img_size[:2], shuffle=shuffle)

        return dataset

    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y

        Arguments:
        * X: The image and text data for which to predict labels. Should be 
          a dataframe with columns "tokens" containing the text and column
          "img_path" containing the full paths to the images
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
        * X: The text and image data for which to cross-validate predictions.
          Should be a dataframe with  text in column "tokens" and image paths
          in column "img_path"
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

        dump(self, os.path.join(save_path, 'model.joblib'))

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

        # Re-building the model and loading the weights which has been saved
        # in model_path
        loaded_model.model, loaded_model.tokenizer, loaded_model.preprocessing_function = loaded_model._getmodel(
            name)

        return loaded_model


class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    MetaClassifier(base_estimators, meta_method='voting', from_trained=None, **kwargs)

    A wrapper that supports various traditional ensemble classifier (voting and stacking for now),
    following the scikit-learn estimator interface.

    Constructor Arguments:
    * base_estimators: list of tuples with the base estimators ([('name', classifier),...]). 
      To exploit the save and load methods, the base estimators should be instances of class 
      TFbertClassifier, MLClassifier, ImgClassifier or TFmultiClassifier. For boosting and 
      bagging methods only the first estimator of the list will be considered.
    * meta_method (str, optional): Ensemble classifier method. Default is 'voting'.
    * from_trained (optional): Path to previously saved model. Default is None.
    * **kwargs: arguments accepted by the chosen sklearn ensemble classifier sepcified by
        meta_method

    Methods:
    * fit(X, y): Trains the model on the provided dataset.
    * predict(X): Predicts the class labels for the given input.
    * predict_proba(X): Predicts class probabilities for the given input (if predict_proba
      is available for the chosen classifier).
    * classification_score(X, y): Calculates weigthed f1-score for the given input and labels.
    * cross_validate(X, y, cv=10): Calculate cross-validated scores with sklearn cross_validate function.
    * save(name): Saves the model to the directory specified in config.path_to_models.

    Example usage:
    meta_classifier = MetaClassifier(base_estimators=[(str1, clf1)], meta_method = 'voting', voting='soft', weight=[0.5, 0.5])
    meta_classifier.fit(train_texts, train_labels)
    predictions = meta_classifier.predict(test_texts)
    f1score = meta_classifier.classification_score(test_texts, test_labels)
    meta_classifier.save('my_ensemble_model')

    """

    def __init__(self, base_estimators, meta_method='voting', from_trained=None, **kwargs):
        """
        Constructor: __init__(self, base_estimators, meta_method='voting', from_trained=None, **kwargs)
        Initializes a new instance of the MetaClassifier.

        Arguments:
        * base_estimators: list of tuples with the base estimators ([('name', classifier),...]). 
          To exploit the save and load methods, the base estimators should be instances of class 
          TFbertClassifier, MLClassifier, ImgClassifier or TFmultiClassifier
        * meta_method (str, optional): Ensemble classifier method. Default is 'voting'.
        * from_trained (optional): Path to previously saved model. Default is None.
        * **kwargs: arguments accepted by the chosen sklearn ensemble classifier sepcified by
            meta_method

        Functionality:
        Initializes the classifier based on the specified meta_method and base_estimators and prepares the model 
        for training or inference as specified.
        """
        self.meta_method = meta_method
        self.from_trained = from_trained
        self.base_estimators = base_estimators
        self.kwargs = kwargs

        if self.from_trained is not None:
            # loading previously saved model if provided
            self = self.load(self.from_trained)
        else:
            # Initialize the model according to base_name and kwargs
            if meta_method.lower() == 'voting':
                self.model = VotingClassifier(base_estimators, **kwargs)
            elif meta_method.lower() == 'stacking':
                self.model = StackingClassifier(base_estimators, **kwargs)
            elif meta_method.lower() == 'bagging':
                self.model = BaggingClassifier(
                    estimator=base_estimators[0][1], **kwargs)
            elif meta_method.lower() == 'boosting':
                self.model = AdaBoostClassifier(
                    estimator=base_estimators[0][1], **kwargs)

            # model_params = self.model.get_params()
            # for param, value in model_params.items():
            #     setattr(self, param, value)

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
        * X: The input data used for training. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens" and/or image paths in column
          "img_path"
        * y: The target labels for training.

        Returns:
        * The instance of MLClassifier after training.
        """
        start_time = time.time()
        self.model.fit(X, y)
        self.fit_time = time.time() - start_time

        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Arguments:
       * X: The input data to use for prediction. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens" and/or image paths in column
          "img_path"

        Returns:
        An array of predicted class labels.
        """
        pred = self.model.predict(X)
        return pred

    def _predict_proba(self, X):
        """
        Predicts class probabilities for the given text data, if the underlying 
        model supports probability predictions.

        Arguments:
        * X: The input data to use for prediction. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens" and/or image paths in column
          "img_path"

        Returns:
        An array of class probabilities for each input instance.
        """
        probs = self.model.predict_proba(X)

        return probs

    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y

        Arguments:
        * X: The text data for which to predict classes. Can be an array like 
          a pandas series, or a dataframe with text in column "tokens" and/or 
          image paths in column "img_path"
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
        * X: The text and image data for which to cross-validate predictions.
          Should be a dataframe with  text in column "tokens" and image paths
          in column "img_path"
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

        # First saving all base estimators in subfolders
        for k, clf in enumerate(self.model.estimators_):
            if isinstance(clf, tuple):
                clf[1].save(name + os.sep + self.base_estimators[k][0])
            else:
                clf.save(name + os.sep + self.base_estimators[k][0])

        # Removing base estimators to save the meta classifier alone (this is
        # because joblib does not serialize keras objects)
        estimators_backup_ = self.model.estimators_
        if isinstance(self.model.estimators_[0], tuple):
            for k in range(len(self.model.estimators_)):
                self.model.estimators_[k] = (
                    self.model.estimators_[k][0], None)
        else:
            self.model.estimators_ = list(range(len(self.model.estimators_)))

        estimators_backup = self.model.estimators
        if isinstance(self.model.estimators[0], tuple):
            for k in range(len(self.model.estimators)):
                self.model.estimators[k] = (self.model.estimators[k][0], None)

        named_estimators_backup_ = self.model.named_estimators_
        for kname in self.model.named_estimators_.keys():
            self.model.named_estimators_[kname] = None

        base_estimators_backup = self.base_estimators
        if isinstance(self.base_estimators[0], tuple):
            for k in range(len(self.base_estimators)):
                self.base_estimators[k] = (self.base_estimators[k][0], None)

        # Saving meta classifier to that location
        dump(self, os.path.join(save_path, 'model.joblib'))

        # Restoring back the base estimators
        self.model.estimators_ = estimators_backup_
        self.model.estimators = estimators_backup
        self.model.named_estimators_ = named_estimators_backup_
        self.base_estimators = base_estimators_backup

    def load(self, name):
        """
        Loads a model from the directory specified in notebook.config file (config.path_to_models).

        Arguments:
        * name: The name of the saved model to load.
        """
        # path to the directory where the model to load was saved
        model_path = os.path.join(
            config.path_to_models, 'trained_models', name)

        # Loading the full model from there
        loaded_model = load(os.path.join(model_path, 'model.joblib'))

        # Loading all base estimators from the  subfolders
        if isinstance(loaded_model.model.estimators_[0], tuple):
            for k, clf in enumerate(loaded_model.model.estimators_):
                base_model = load(os.path.join(
                    model_path, clf[0], 'model.joblib'))
                base_model.load(name + os.sep + clf[0])
                loaded_model.model.estimators_[k] = (clf[0], base_model)
        else:
            for k in range(len(loaded_model.model.estimators_)):
                base_model = load(os.path.join(
                    model_path, str(k), 'model.joblib'))
                base_model.load(name + os.sep + str(k))
                loaded_model.model.estimators_[k] = base_model

        return loaded_model
