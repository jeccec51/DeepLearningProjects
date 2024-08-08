"""Module to download mot 20 data set."""

import os
import requests
import zipfile

def download_file(url, destination) -> None:
    """Download a file from a URL.
    
    Args:
        url: URL of the file to download.
        destination: Destination file path.
    """

    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def download_and_extract_mot20(dest_dir: str) -> None:
    """Download and extract the MOT20 dataset.
    
    Args:
        dest_dir: Destination directory to save the dataset.
    """

    mot20_train_url = 'https://motchallenge.net/data/MOT20.zip'

    zip_path = os.path.join(dest_dir, 'MOT20.zip')

    # Download the dataset
    print("Downloading MOT20 dataset...")
    download_file(mot20_train_url, zip_path)

    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    # Remove the zip file
    os.remove(zip_path)

    print("MOT20 dataset downloaded and extracted successfully.")


if __name__ == "__main__":
    dest_dir = './data'
    os.makedirs(dest_dir, exist_ok=True)
    download_and_extract_mot20(dest_dir)
