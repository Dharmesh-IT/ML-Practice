import os
import tarfile
import urllib
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/blob/58e00c65958dc4f51024adb41db40a3b4111c755/datasets/housing/"
HOUSING_PATH=os.path.join('datasets','housing')
HOUSING_URL = DOWNLOAD_ROOT + "housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path="datasets/housing/housing.csv"
    return pd.read_csv(csv_path)


housing=load_housing_data()
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50,figsize=(20,15))
plt.show()