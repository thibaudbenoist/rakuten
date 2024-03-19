import config
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from src.data_loader import import_data
from sklearn.pipeline import Pipeline
from src.features.text.transformers.cleaners import HtmlCleaner, LxmlCleaner, TextCleaner
from src.features.text.transformers.text_merger import TextMerger
from src.features.text.transformers.translators import TextTranslator
from src.features.text.pipelines.cleaner import CleanTextPipeline
from src.features.text.pipelines.corrector import CleanEncodingPipeline
import time

merger = TextMerger(description_column="description", designation_column="designation", merged_column="merged_text")
cleaner = CleanTextPipeline()
corrector = CleanEncodingPipeline()
translator = TextTranslator(text_column="corrected_text", target_lang="fr")

raw_df = import_data(cleaned=False).head(0) 
merged_text = merger.fit_transform(raw_df)
cleaned_text = cleaner.fit_transform(merged_text)
corrected_text = corrector.fit_transform(cleaned_text)
corrected_text = pd.DataFrame(corrected_text.rename("corrected_text"))

translated_text = translator.fit_transform(corrected_text)


# Sélection des colonnes à afficher
selected_columns = ['designation', 'description']

# Affichage d'un extrait du DataFrame brut
st.write("### Extrait du DataFrame Brut:")
st.write(raw_df[selected_columns])


#Text merging
if st.button("Appliquer Text Merger"):
    with st.spinner("Merging en cours"):
        time.sleep(1)
        st.success("TextMerger appliqué avec succès !")
        st.write(merged_text)


if st.button("Appliquer le cleaning du texte"):
    with st.spinner("Cleaning en cours"):
        time.sleep(2)
        st.success("Text Markups cleaning appliqué avec succès !")
        st.write(cleaned_text)

if st.button("Appliquer la correction de l'encoding"):
    with st.spinner("Cleaning en cours"):
        time.sleep(2)
        st.success("Correction d'encoding appliquée avec succès !")
        st.write(corrected_text)

if st.button("Appliquer la traduction"):
    with st.spinner("Traduction en cours"):
        time.sleep(2)
        st.success("Traduction appliquée avec succès !")
        st.write(translated_text)




