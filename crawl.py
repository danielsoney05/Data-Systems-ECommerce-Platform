import kagglehub
import shutil
import os

path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Path to dataset files:", path)

destination = "data/"

for file in os.listdir(path):
    shutil.copy(os.path.join(path, file), destination)

print("Files copied to data/")