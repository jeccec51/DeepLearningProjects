"""Module to download mot 20 data set."""

import os
import requests
import zipfile
from tqdm import tqdm


def download_file(url, destination) -> None:
    """Download a file from a URL with a progress bar.
    
    Args:
        url: URL of the file to download.
        destination: Destination file path.
    """

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    with open(destination, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc="Downloading", 
        unit_divisor=1024, leave=True
    ) as bar:
        for chunk in response.iter_content(block_size):
            bar.update(len(chunk))
            f.write(chunk)


def extract_file(zip_path: str, dest_dir: str) -> None:
    """Extract a zip file with a progress bar.
    
    Args:
        zip_path: Path to the zip file.
        dest_dir: Directory to extract the contents.
    """

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, unit='file', desc="Extracting") as bar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, dest_dir)
                bar.update(1)


def download_and_extract_mot20(dest_dir: str) -> None:
    """Download and extract the MOT20 dataset.
    
    Args:
        dest_dir: Destination directory to save the dataset.
    """

    mot20_train_url = 'https://motchallenge.net/data/MOT20.zip'

    zip_path = os.path.join(dest_dir, 'MOT20.zip')

    # Download the dataset
    download_file(mot20_train_url, zip_path)

    # Extract the dataset
    extract_file(zip_path, dest_dir)

    # Remove the zip file
    os.remove(zip_path)

    print("MOT20 dataset downloaded and extracted successfully.")


if __name__ == "__main__":
    dest_dir = './data'
    os.makedirs(dest_dir, exist_ok=True)
    download_and_extract_mot20(dest_dir)
