import os
from pathlib import Path
import sys
sys.path.append("C:/Users/rim18/OneDrive/Documentos/GitHub/Final-Project-Anyone-AI---Automated-product-categorization-for-e-commerce-with-AI")
# Determina si estás en Google Colab
# IN_COLAB = 'google.colab' in str(get_ipython())

# if IN_COLAB:
#     DATASET_ROOT_PATH = "/content/drive/MyDrive/Automated product categorization for e-commerce with AI/dataset/"
# else:
#     DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")

# # Rutas de los archivos descargados
# PRODUCTS = os.path.join(DATASET_ROOT_PATH, "products.json")
# CATEGORIES = os.path.join(DATASET_ROOT_PATH, "categories.json")

# # URLs de los datasets en GitHub (raw URLs)
DATASET_PRODUCTS_URL = "https://raw.githubusercontent.com/anyoneai/e-commerce-open-data-set/master/products.json"
DATASET_CATEGORIES_URL = "https://raw.githubusercontent.com/anyoneai/e-commerce-open-data-set/master/categories.json"

#Local

import requests

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# Create the dataset folder if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Function to download and save a JSON file
def download_and_save_json(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(dataset_path, file_name)
        with open(file_path, 'w') as file:
            file.write(response.text)
        print(f"File saved at: {file_path}")
    else:
        print(f"Error downloading the file from {url}")
