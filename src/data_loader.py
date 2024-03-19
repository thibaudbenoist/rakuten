import sys
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from sklearn.base import BaseEstimator, TransformerMixin


def get_base_dir():
    load_dotenv()
    BASE_DIR = os.getcwd()
    DIR_NAME = os.path.basename(BASE_DIR)
    while DIR_NAME != os.environ.get("PROJECT_NAME"): 
        BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__name__), '..'))
        DIR_NAME = os.path.basename(BASE_DIR)
        os.chdir(BASE_DIR)
        DATA_DIR = os.path.join(BASE_DIR, 'data')
    return BASE_DIR
    
    

if __name__ == "__main__":
    get_base_dir()




def import_data(cleaned=True):
    """
    Importe les données nécessaires à l'entraînement ou à la prédiction d'un modèle.

    Args:
        cleaned (bool): Si True, les données nettoyées sont importées, sinon les données brutes sont utilisées. Par défaut, True.

    Returns:
        dict or DataFrame: Un dictionnaire contenant les données d'entraînement et de test si `cleaned` est True.
                           Sinon, un DataFrame contenant les données d'entraînement et la variable cible.

    Exemple:
        Pour importer les données nettoyées :
        >>> data = import_data(cleaned=True)

        Pour importer les données brutes :
        >>> data = import_data(cleaned=False)
        Note: les deux clés de l'output de "cleaned=True" sont "train" et "test"
    """
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    clean_data_dir = os.path.join(data_dir, "clean")
    if not cleaned:
        folder_path = raw_data_dir
        data = pd.read_csv(os.path.join(folder_path, 'X_train.csv'), index_col=0)
        target = pd.read_csv(os.path.join(folder_path, 'Y_train.csv'), index_col=0)
        data = target.join(data)
    else:     
        folder_path = clean_data_dir
        df_train = pd.read_csv(os.path.join(folder_path, "df_train_index.csv"), index_col=0)
        df_test = pd.read_csv(os.path.join(folder_path, "df_test_index.csv"), index_col=0)
        data = {"train": df_train, "test": df_test}
    return data


class CatNameExtractor(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, data_dir) -> None:
        self.mapper = pd.read_csv(os.path.join(data_dir, "prdtype.csv")).set_index(
            keys="prdtypecode").prdtypedesignation.to_dict()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(self.mapper)
        X["prdtypename"] = X["prdtypecode"].map(self.mapper)
        return X


class ClassMerger(BaseEstimator, TransformerMixin):
    """
    WIP : not working yet used for testing a system based on simple rules
    """

    def __init__(self) -> None:
        self.mapper = {
            10: 2403,
            2705: 2403,
            2280: 2403,
            2905: 40,
            2462: 40
        }


if __name__ == "__main__":
    data = import_data(cleaned=True)
    print(data.get("train"))