from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
import tensorflow as tf

import numpy as np
import pandas as pd
import cv2

from vit_keras import vit, utils, visualize, layers
from transformers import TFAutoModel, AutoTokenizer, CamembertTokenizer, CamembertModel, FlaubertTokenizer, CamembertConfig
from wordcloud import WordCloud

import os
import sys
import matplotlib.pyplot as plt

import config

from src.text.classifiers import TFbertClassifier
from src.image.classifiers import ImgClassifier
from src.multimodal.classifiers import TFmultiClassifier

class deepCAM:
    """
    A class for generating class activation maps and attention maps for images and text using
    deep learning models. This class supports handling both image and text inputs, utilizing
    specific layers and attention mechanisms of the underlying models to visualize areas of
    interest or focus.

    Attributes:
        classifier (object): A classifier model that is either an image classifier, a text
                             classifier, or a multi-modal classifier that handles both image
                             and text inputs.
        img_base_name (str): The base name for the image classifier model, used to determine
                             specific processing based on the model type.
        txt_base_name (str): The base name for the text classifier model, used similarly to
                             `img_base_name`.
        preprocessing_function_img (callable): The preprocessing function for images, extracted
                                               from the classifier if available.
        preprocessing_function_txt (callable): The preprocessing function for text, to be defined
                                               based on the classifier type.
        decode_function_txt (callable): A function to decode token IDs back to text, relevant for
                                        text models.
        img_size (tuple): The expected input size for images, derived from the classifier.
        gradModel_img (tf.keras.Model): A gradient model for computing gradients with respect to
                                        image inputs, relevant for CAM visualization.
        gradModel_txt (object): A model or mechanism for computing attention or gradients with
                                respect to text inputs.

    Methods:
        grad_map(image, min_factor=0.0):
            Generates a gradient-based class activation map for an image input.

        attention_map_img(image, min_factor=0.0, last_attn_only=False):
            Generates an attention map for an image input using Vision Transformer (ViT) models.

        attention_map_txt(text, min_factor=0.0, last_attn_only=False):
            Generates an attention map for text input using models like BERT.

        computeMaskedInput(X, min_factor=0.0, last_attn_only=True):
            A convenience method to compute and return masked inputs (image and/or text) based
            on the model's focus areas or important features.
    """
    def __init__(self, classifier):
        """
        Initializes a deepCAM instance with a given classifier and sets up preprocessing functions
        and models for generating class activation maps and attention maps based on the type of
        classifier provided.
        
        Parameters:
            classifier (object): The classifier model. This could be an image classifier,
                                 a text classifier, or a multi-modal classifier handling both.
        """
        self.classifier = classifier
        self.img_base_name = 'None'
        self.txt_base_name = 'None'
        self.preprocessing_function_img = None
        self.preprocessing_function_txt = None
        self.decode_function_txt = None
        self.img_size = None
        
        if isinstance(classifier, ImgClassifier):
            self.preprocessing_function_img = classifier.preprocessing_function
            self.img_size = classifier.img_size
            self.img_base_name = classifier.base_name
        elif isinstance(classifier, TFbertClassifier):
            self.txt_base_name = classifier.base_name
        elif isinstance(classifier, TFmultiClassifier):
            self.preprocessing_function_img = classifier.preprocessing_function
            self.img_size = classifier.img_size
            self.txt_base_name = classifier.txt_base_name
            self.img_base_name = classifier.img_base_name
        
        if 'resnet' in self.img_base_name.lower():
            x = classifier.model.get_layer('img_base_layers')(classifier.model.inputs)
            lastconv_outputs = Dropout(rate=0, name='dummy_layer')(x)
            x = lastconv_outputs
            
            is_top_layer = False
            for layer in classifier.model.layers:
                if is_top_layer:
                    x = layer(x)
                elif layer.name == 'img_base_layers':
                    is_top_layer = True
                    
            outputs = x

            self.gradModel_img = Model(
                        inputs=[classifier.model.inputs],
                        outputs=[lastconv_outputs, outputs])
            
        if 'vit' in self.img_base_name.lower():
            self.gradModel_img = vit.vit_b16(image_size=classifier.img_size[0:2], pretrained=True, include_top=False, pretrained_top=False)
            for grad_layer, orig_layer in zip(self.gradModel_img.layers[:-2], classifier.model.get_layer('img_base_layers').layers):
                grad_layer.set_weights(orig_layer.weights)
                
        if 'bert' in self.txt_base_name.lower():
            base_model_path = os.path.join(config.path_to_models, 'base_models', self.txt_base_name)
            if 'camembert' in self.txt_base_name.lower():
                bert_config = CamembertConfig.from_pretrained('camembert-base', output_attentions=True, output_hidden_states=False)
                base_model = TFAutoModel.from_pretrained(base_model_path, config=bert_config)
                tokenizer = CamembertTokenizer.from_pretrained(base_model_path)
            elif 'flaubert' in self.txt_base_name.lower():
                base_model = TFAutoModel.from_pretrained(base_model_path)
                tokenizer = FlaubertTokenizer.from_pretrained(
                    'flaubert/flaubert_base_uncased')
                
            self.preprocessing_function_txt = lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=classifier.max_length, return_tensors="tf")
            self.decode_function_txt = lambda x: tokenizer.convert_ids_to_tokens(x)
            self.gradModel_txt = base_model
            for grad_layer, orig_layer in zip(self.gradModel_txt.layers[0].encoder.layer, classifier.model.get_layer('txt_base_layers').layers[0].encoder.layer):
                grad_layer.set_weights(orig_layer.weights)
          
          
          
                
    def grad_map(self, image, min_factor=0.0):
        """
        Generates a gradient-based class activation map (CAM) for an image input. This method
        is applicable for models like ResNet where gradient information with respect to a specific
        convolutional layer can highlight important regions in the image for a particular class.

        Parameters:
            image (np.ndarray): The input image for which the CAM is to be generated.
            min_factor (float, optional): A threshold value to enhance visibility of the CAM.
                                          Defaults to 0.0.

        Returns:
            np.ndarray: The image masked with the generated CAM, highlighting regions of interest.
        """        
        image_orig = cv2.resize(image, self.img_size[:2])
        image = self.preprocessing_function_img(image_orig)
        image = image.reshape((1,) + image.shape)
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            # Ensure this call is within the tape context to track operations
            (convOutputs, predictions) = self.gradModel_img(inputs)
            tape.watch(convOutputs)
            # Assuming your model's final layer uses a softmax activation
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Now, compute the gradients of the loss w.r.t the convOutputs
        grads = tape.gradient(loss, convOutputs)
        
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        mask = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        eps=1e-8
        mask = (mask - np.min(mask)) / ((mask.max() - mask.min()) + eps)
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
        
        # save mask, original image and image overlaid with mask
        self.mask_img = mask
        self.image = image_orig
        mask[mask < min_factor] = min_factor
        self.image_masked = (mask * self.image).astype("uint8")
        
        return self.image_masked
    
    
    
    def attention_map_img(self, image, min_factor=0.0, last_attn_only=False):
        """
        Generates an attention map for an image input using Vision Transformer (ViT) models.
        This method visualizes the attention weights from the transformer blocks to highlight
        regions of interest in the input image.

        Parameters:
            image (np.ndarray): The input image for which the attention map is to be generated.
            min_factor (float, optional): A threshold value to enhance visibility of the attention map.
                                          Defaults to 0.0.
            last_attn_only (bool, optional): Whether to use only the last attention layer's weights.
                                             Defaults to False.

        Returns:
            np.ndarray: The image masked with the generated attention map.
        """
        img_height, img_width = self.gradModel_img.input_shape[1], self.gradModel_img.input_shape[2]
        grid_size = int(np.sqrt(self.gradModel_img.layers[5].output_shape[0][-2] - 1))

        # Prepare the input
        image_orig = cv2.resize(image, (img_height, img_width))
        X = self.preprocessing_function_img(image_orig)[np.newaxis, :]

        # Get the attention weights from each transformer.
        outputs = [layer.output[1] for layer in self.gradModel_img.layers if isinstance(layer, layers.TransformerBlock)]
        weights = np.array(tf.keras.models.Model(inputs=self.gradModel_img.inputs, outputs=outputs).predict(X))
        num_layers = weights.shape[0]
        num_heads = weights.shape[2]
        reshaped = weights.reshape((num_layers, num_heads, grid_size**2 + 1, grid_size**2 + 1))

        # From Appendix D.6 in the paper ...
        # Average the attention weights across all heads.
        reshaped = reshaped.mean(axis=1)

        # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        reshaped = reshaped + np.eye(reshaped.shape[1])
        reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

        # Recursively multiply the weight matrices
        v = reshaped[-1]
        if not last_attn_only:
            for n in range(1, len(reshaped)):
                v = np.matmul(v, reshaped[-1 - n])

        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size)
        mask = (mask / mask.max())
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
        mask = cv2.resize(mask, image_orig.shape[:2], interpolation = cv2.INTER_LINEAR)
        
        self.mask_img = mask
        self.image = image_orig
        mask[mask < min_factor] = min_factor
        self.image_masked = (mask * self.image).astype("uint8")
        
        return self.image_masked
    
    def attention_map_txt(self, text, min_factor=0.0, last_attn_only=False):
        """
        Generates an attention map for text input using models like BERT. This method visualizes
        the attention weights to highlight important words or tokens in the input text.

        Parameters:
            text (str or list): The input text for which the attention map is to be generated.
            min_factor (float, optional): A threshold value to enhance the visibility of the attention map.
                                          Defaults to 0.0.
            last_attn_only (bool, optional): Whether to use only the last attention layer's weights.
                                             Defaults to False.

        Returns:
            np.ndarray: An array representing the attention weights for each token in the
        """
        X = self.preprocessing_function_txt(text)
        seq_length = X['input_ids'].shape[1]
        
        # Get the attention weights from each transformer.
        outputs = self.gradModel_txt(X)
        num_layers = len(outputs.attentions)
        for k in range(num_layers):
            weights = outputs.attentions[k].numpy()
            num_heads = weights.shape[1]
            weights = weights.reshape((1, num_heads, seq_length, seq_length))
            if k == 0:
                reshaped = weights
            else:
                reshaped = np.concatenate([reshaped, weights], axis=0)

        # From Appendix D.6 in the paper ...
        # Average the attention weights across all heads.
        reshaped = reshaped.mean(axis=1)

        # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        reshaped = reshaped + np.eye(reshaped.shape[1])
        reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

        # Recursively multiply the weight matrices
        v = reshaped[-1]
        if not last_attn_only:
            for n in range(1, len(reshaped)):
                v = np.matmul(v, reshaped[-1 - n])

        # Attention from the output token to the input space.
        mask = v[0, 1:]
        
        #Selecting tokens with non-zeros weights (=non-padded)
        valid_mask_idx = np.argwhere(mask > 0)
        valid_mask_idx = np.squeeze(valid_mask_idx).tolist()
        #Removing last <sep> token
        valid_mask_idx.pop(-1)
        
        mask = (mask[valid_mask_idx] / mask[valid_mask_idx].max())
        self.mask_txt = mask
        
        allwords = self.decode_function_txt(X['input_ids'][0][1:])
        allwords = [word for idx, word in enumerate(allwords) if idx in valid_mask_idx]
        word_length = np.array([len(word) for word in allwords])
        min_mask = self.mask_txt[word_length < 3].min()
        self.mask_txt[word_length < 3] = min_mask
        word_orig = [word.replace('▁', ' ') for word in allwords]
        # word_bert = [word.replace('▁', ' ') for word in enumerate(allwords)]

        self.text = word_orig
        self.text_masked = self.mask_txt
        
        return self.text_masked
    
    
    
    def computeMaskedInput(self, X, min_factor=0.0, last_attn_only=True):
        """
        Computes and returns masked inputs for both images and text, highlighting areas or features
        deemed important by the model. This method serves as a convenience wrapper to apply the
        appropriate visualization technique (CAM or attention map) based on the input type and
        the underlying model.

        For image inputs, this method can generate gradient-based CAMs for CNNs or attention maps
        for Vision Transformers (ViT). For text inputs, it generates attention maps using models
        like BERT.

        Parameters:
            X (dict, np.ndarray, or str): The input data for which masked inputs are to be generated.
                                          This can be:
                                          - A dictionary with keys 'image' and 'text' for multi-modal
                                            inputs.
                                          - An np.ndarray for image inputs.
                                          - A string or list of strings for text inputs.
            min_factor (float, optional): A threshold value to enhance the visibility of the masked
                                          areas. Regions with importance scores below this threshold
                                          are deemphasized. Defaults to 0.0.
            last_attn_only (bool, optional): Specifies whether to use only the last attention layer's
                                             weights for generating attention maps. This is relevant
                                             for models like ViT and BERT. Defaults to True.

        Returns:
            tuple: A tuple containing:
                   - The masked image input (np.ndarray or None if not applicable).
                   - The masked text input (np.ndarray or None if not applicable).

        This method facilitates the exploration of model behaviors and decisions by visualizing
        which parts of the inputs are deemed important for the model's predictions. It automatically
        chooses the appropriate visualization method based on the input type and the characteristics
        of the provided model.
        """
        image_masked = None
        text_masked = None
        
        if isinstance(X, dict):
            X_img = X['image']
            X_txt = X['text']
        elif isinstance(X, np.ndarray):
            X_img = X
            X_txt = None
        else:
            X_img = None
            X_txt = X
            
        if 'resnet' in self.img_base_name.lower() and X_img is not None:
            image_masked = self.grad_map(X_img, min_factor)
        if 'vit' in self.img_base_name.lower() and X_img is not None:
            image_masked = self.attention_map_img(X_img, min_factor, last_attn_only=False)
        if 'bert' in self.txt_base_name.lower() and X_txt is not None:
            text_masked = self.attention_map_txt(X_txt, min_factor, last_attn_only)
            
        return image_masked, text_masked
    
    
def plot_weighted_text(x, y, words, weights, base_font_size=10, ax=None, char_per_line=10, title='None', title_color='purple', title_fontsize=100, **kw):
    """
    Displays a sequence of words within a matplotlib Axes object, with each word's font size adjusted
    according to its corresponding weight. Additionally, the function supports adding a title with
    customizable font size and color. The text layout is managed by specifying a maximum number of
    characters per line, which influences word wrapping.

    Parameters:
    - x (float): The x-coordinate (in Axes' fraction from 0 to 1) for the start of the text block.
    - y (float): The y-coordinate (in Axes' fraction from 0 to 1) for the start of the text block.
    - words (list of str): A list of words to display within the text block.
    - weights (list of float): A list of weights corresponding to each word in `words`. These weights
      are used to adjust the font size of each word relative to `base_font_size`.
    - base_font_size (int, optional): The base font size for text. Actual font sizes are determined
      by multiplying this value by each word's weight. Defaults to 10.
    - ax (matplotlib.axes.Axes, optional): The matplotlib Axes object to draw the text on. If None,
      a new figure and axes are created. Defaults to None.
    - char_per_line (int, optional): The maximum character length of a line before wrapping to a new
      line. This helps manage the layout of the text block. Defaults to 10.
    - title (str, optional): The title text displayed at the top of the text block. Defaults to 'None'.
    - title_color (str, optional): The color of the title text. Can be any matplotlib-recognized color.
      Defaults to 'purple'.
    - title_fontsize (int, optional): The font size of the title text, which is independent of the
      `base_font_size` and not influenced by the weights list. Defaults to 100.
    - **kw: Additional keyword arguments passed to `plt.text()` for customizing text appearance, such
      as font style or background color.

    The function directly modifies the provided `ax` object by plotting the title (if not 'None') and
    the sequence of words according to the specified layout and style parameters. The `weights` list
    affects the relative size of each word in the text block, allowing for visual emphasis based on
    weight values.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    lines = [[title]]
    weight_segments = [[1/base_font_size * title_fontsize]]
    
    word_line = []
    weight_line = []
    char_length = 0
    for word, weight in zip(words, weights):
        word_line.append(word)
        weight_line.append(weight)
        char_length += len(word)
        if char_length > char_per_line:
            lines.append(word_line)
            weight_segments.append(weight_line)
            char_length = 0
            word_line = []
            weight_line = []
    
    if len(lines) == 1:
        lines.append(word_line)
        weight_segments.append(weight_line)
        
    first_line = True
    color = title_color
    current_y = y
    for line, weight_segment in zip(lines, weight_segments):
        current_x = x
        for word, weight in zip(line, weight_segment):
            # Calculate font size based on weight
            font_size = base_font_size * weight
            # Display each word with the adjusted font size
            text_handle = plt.text(current_x, current_y, word, fontsize=font_size, color=color, transform=ax.transAxes, **kw)
            # Measure the text size to adjust the next word's position
            text_extent = text_handle.get_window_extent()
            current_x += 0.2*text_extent.width / fig.dpi  # Adjust for next word position
        if first_line:
            current_y -= (0.4*base_font_size / fig.dpi)  # Move to next line
            first_line = False
            color = 'black'
        else:
            current_y -= (0.2*base_font_size / fig.dpi)  # Move to next line

    current_y -= (0.6*base_font_size / fig.dpi)
    
    ax.axis('off');
    ax.set_xlim(0, 1);
    ax.set_ylim(current_y, y);