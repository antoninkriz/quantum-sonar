import io
import os
import zipfile

import pandas as pd
import requests


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
