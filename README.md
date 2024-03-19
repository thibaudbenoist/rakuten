# DataScientest - RakutenTeam
- [Thibaud BENOIST](https://www.linkedin.com/in/thibaud-benoist-76593730/)
- [Julien CHANSON](https://www.linkedin.com/in/julienchanson/)
- [Julien FOURNIER](https://www.linkedin.com/in/julien-fournier-63530537/)
- [Alexandre MANGWA](https://www.linkedin.com/in/alexandre-mangwa-7a3aa1140/)

# Rakuten Challenge

In this project, the objective was to evaluate the capability of various classification methods to categorize text and image-based products from a marketplace catalog. 
Specialized models handling text and images separately were first benchmarked before being merged into hybrid classifiers. 

The combination by simple majority vote of the hybrid model (a transformer merging BERT and ViT) with several specialized models (BERT and XGBoost for text; ViT and ResNet for images) yielded a weighted-F1 score of **0.911** after training on 80% of the data. 

This score is highly satisfactory and ranks among the highest scores in both public and private leaderboards on the challenge site from which this dataset originates.

This project was carried out as part of our data scientist training with the [DataScientest](https://datascientest.com/) training organization, from December 2023 to March 2024.

## Repository overview
```
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── data                            <- Should be in your computer but not on Github (only in .gitignore)
│   ├── clean                       <- Contains dataset cleaned after preprocessing
│   │   └── df_text_index.csv       <- Dataset 'text' used for models testing, represents 20% of original dataset
│   │   └── df_train_index.csv      <- Dataset 'text' used for models training, represents 80% of original dataset
│   │   └── le_classes.npy          <- Numpy array of label encoder classes 
│   ├── images                      <- Directory containing the image (to be download on challenge site)
│   └── raw                         <- The original, immutable data dump.
│
├── models                          <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                       <- Jupyter notebooks
│
├── reports                         <- The PDF report
│
├── requirements.txt                <- The requirements file for pip
|
├── env_wsl.yml                         <- The conda yaml environnment file
│
├── src                             <- Source code for use in this project.

```

## Quick Start

### Installation
1- Deploy the python environment :
```bash
conda create -p ./env -f env_wsl.yml
```
or 

```bash
pip install -r requirements.txt
```

2- Download images and dataset on Rakuten challenge site, unzip images 

```
wget -O data/raw/X_train.csv https://challengedata.ens.fr/participants/challenges/35/download/x-train
wget -O data/raw/y_train.csv https://challengedata.ens.fr/participants/challenges/35/download/y-train
wget -O data/raw/X_test.csv https://challengedata.ens.fr/participants/challenges/35/download/x-test
wget -O data/images/images.zip https://challengedata.ens.fr/participants/challenges/35/download/supplementary-files
```

3- Download the already cleaned data via this link and unzip the three files into `/data/clean`.
[Data](https://drive.google.com/file/d/19m9KGL0YJoQgC1kOm4yhQODdXK9sKxcu/view?usp=sharing)

4- Download the training results using the following link and unzip the .csv files into `/results`.
[Results](https://drive.google.com/file/d/1YWTlxLgz3T34OtwSi3p7Q2f4CcYIdDU7/view?usp=sharing)

5- Download the trained models to predict new datas (used for streamlit predictions)
[Models](https://drive.google.com/drive/folders/1-M6AwQJ82Uhwd-T8ljocdvSQBj_i3zwl?usp=sharing)

6- Create or edit the config files located in `notebook/config.py` and `streamlit/config.py` to specify the local paths of the project folder.
```python
import os
import sys
dir_root = '/path/to/my/root/folder'

project_dir = '/path/to/my/project/folder'
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'src'))

# directory of the project
path_to_project = project_dir
# path to the data (dataframe)
path_to_data = project_dir + '/data/clean'
# path to where the summary of the benchmark results will be saved (csv)
path_to_results = project_dir + '/results'
# path to the folder containing images
path_to_images = project_dir + '/images/image_train_resized'
# path to the folder where the models will be saved
path_to_models = project_dir + '/models'
# Path to where tensorboard logs will be saved
path_to_tflogs = project_dir + '/tflogs'
```

### Preprocessing

For launching CleanTextPipeline
```python
from src.features.text.pipelines.cleaner import CleanTextPipeline
pipe = CleanTextPipeline()
cleaned_descriptions = pipe.fit_transform(descriptions)
```

For launching CleanDescription
```python
from src.features.text.pipelines.corrector import CleanEncodingPipeline
pipe = CleanEncodingPipeline()
corrected_descriptions = pipe.fit_transform(cleaned_descriptions)
```

For translation
```python
from src.features.text.transformers.translators import TextTranslator

translator = TextTranslator(
    text_column="designation",
    lang_column="language"
)

print(translator.lang_column, translator.text_column, translator.target_lang)

translations = translator.fit_transform(df)
```

For preprocessing images
```python
from src.features.image.functions.resizer import img_resize
img_resize('/data/images')
```

For getting image path from dataset
```python
from src.features.images.transformers.path_finder import PathFinder

finder = PathFinder(img_suffix="_resized")
df["path"] = finder.fit_transform(df)
```

### Training models

cf. 
- `notebook/notebook_benchmark_txt_bert.ipynb`
- `notebook/notebook_benchmark_fusion_meta.ipynb`
- `notebook/notebook_benchmark_fusion_TF.ipynb`
- `notebook/notebook_benchmark_img.ipynb`
- `notebook/notebook_benchmark_txt_ML.ipynb`

### Report 
cf.
- `reports/report_figures.ipynb`


## Code architecture

### Feature preprocessors
All features classes are based on sklearn Transformers pattern and organized by function
The code is in `src/features` directory
```
├── images                          <- images preprocessors
├── text                            <- Should be in your computer but not on Github (only in .gitignore)
│   ├── pipelines                   <- Combinations 
│   │   └── cleaner.py              <- Pipeline for basic text cleaning
│   │   └── corrector.py            <- Pipeline for regex cleaning
│   ├── transformers                <- Directory containing all the text transformers
│   │   └── transformer_xxx.py      <- transformer

```

the `vizualisation/wordclouds.py` file contains methods far drawing wordcloud images.

### Classifiers and results parsers
```
├── image                           <- images classifiers
│   ├── classifiers.py              <- file containing ImgClassifier class
├── multimodal                      <- multimodal classifiers
│   ├── classifiers.py              <- file containing MetaClassifier and TFmultiClassifier classes
├── text                            <- text classifiers
│   ├── classifiers.py              <- file containing MLClassifier and TFbertClassifier classes
│   ├── vectorizers.py              <- file containing Word2Vec vectorizers transformers 
├── utils                           <- text classifiers
│   ├── batch.py                    <- file containing `fit_save_all()` method for orchestrating all benchmarks
│   ├── load.py                     <- file for loading and saving fitted models
│   ├── plot.py                     <- helper to plot 
│   ├── results.py                  <- file containing ResultsManager class
```


There are five classes for different aspects of the multimodal classification task:
- `TFbertClassifier`: For text classification using BERT models.
- `MLClassifier`: For classification tasks using traditional machine learning algorithms.
- `ImgClassifier`: For image classification tasks using pre-trained models like Vision Transformer (ViT), EfficientNet, ResNet, VGG16, and VGG19.
- `TFmultiClassifier`: For multimodal deep networks that combines text and image data.
- `MetaClassifier`: For applying ensemble methods (voting, stacking, bagging, and boosting) to improve model performance by combining multiple of the above classifiers.
