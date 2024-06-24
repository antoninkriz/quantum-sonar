import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


DATA_URL = 'https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip'
CACHED_DATA_FILE = os.path.join('data', 'sonar.all-data')


def load_data_from_internet(cache_file_path: str = CACHED_DATA_FILE) -> pd.DataFrame:
    '''
    Dowloads the Sonar Mines vs Rocks dataset into cache_file_path and loads it to pandas DataFrame.
    Feature columns are called f0, f1, ... f59 and the target is in the Y column.
    https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
    '''
    cahed_file_folder = os.path.abspath(os.path.join(CACHED_DATA_FILE, os.pardir))

    if not os.path.exists(cahed_file_folder):
        os.makedirs(cahed_file_folder, exist_ok=True)

    if not os.path.isfile(cache_file_path):
        response = requests.get(DATA_URL)
        response.raise_for_status()

        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        with zip_file.open('sonar.all-data') as zip_content_file:
            contents = zip_content_file.read()

            with open(cache_file_path, 'wb') as cache_file:
                cache_file.write(contents)

    return pd.read_csv(cache_file_path, header=None, names=[f'f{i}' for i in range(60)] + ['Y'])


def scale_and_split(dataframe: pd.DataFrame, test_size: float = 0.2, pca_components: int = None, random_state: int = 1337) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns X_train, X_test, y_train, y_test after scaling the features and splitting the data.

    X_train and X_test are is numpy arrays with the features scaled using StandardScaler as float64.
    Y_train and Y_test are numpy arrays with the target values as boolean. True for 'M'ines and False for 'R'ocks.
    '''
    X = dataframe.drop(columns=['Y'])
    y = dataframe['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train == 'M', y_test == 'M'
