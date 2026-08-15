import kagglehub
from kagglehub import KaggleDatasetAdapter
import shutil

# Set the path to the file you'd like to load
file_path = "backend/src/notebooks/data/input_images"

# Load the latest version
download_path = kagglehub.dataset_download(
    "iamsouravbanerjee/indian-food-images-dataset"
)

shutil.copytree(download_path, file_path, dirs_exist_ok=True)

print("Done")