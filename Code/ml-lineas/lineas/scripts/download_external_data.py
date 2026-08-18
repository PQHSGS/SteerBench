# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def download_file(url: str, filename: Path, hf_token: str = None) -> None:
    """
    Downlaods a file and saves it as `filename`

    :param url: The url of the zip file
    :param filename: Destination file
    :param hf_token: HF token
    """
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None
    response = requests.get(url, headers=headers, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    if response.status_code == 200:
        with (
            open(filename, "wb") as file,
            tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            for chunk in response.iter_content(1024):
                file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"File downloaded as '{filename}'")
    else:
        print("Failed to download file")


def download_and_extract_zip(url, extract_to=".", hf_token: str = None) -> None:
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None

    # Step 1: Send a GET request to start the download
    response = requests.get(url, headers=headers, stream=True)

    # Step 2: Check if the request was successful
    if response.status_code == 200:
        # Get the total file size from the headers, if available
        total_size = int(response.headers.get("content-length", 0))

        # Step 3: Initialize the progress bar
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as progress_bar:
            # Step 4: Download the file in chunks and update the progress bar
            file_bytes = io.BytesIO()
            for chunk in response.iter_content(chunk_size=1024):
                file_bytes.write(chunk)
                progress_bar.update(len(chunk))

            # Step 5: Load the zip file from memory and extract
            zip_file = zipfile.ZipFile(file_bytes)
            zip_file.extractall(path=extract_to)
            print(f"Files extracted to {extract_to}")
    else:
        print("Failed to download the file:", response.status_code)


if __name__ == "__main__":
    data_dir = Path(os.environ.get("DATA_DIR", "/tmp/lineas-data"))
    data_dir.mkdir(exist_ok=True, parents=True)

    token = os.getenv("HF_TOKEN", None)
    assert token is not None, "HF_TOKEN environment variable is not set"

    # TET
    url = "https://huggingface.co/datasets/convoicon/Thoroughly_Engineered_Toxicity/resolve/main/data/train-00000-of-00001.parquet"
    download_file(
        url, data_dir / "thoroughly_engineered_toxicity.parquet", hf_token=token
    )
    # Load the parquet file and save as JSONL
    df = pd.read_parquet(data_dir / "thoroughly_engineered_toxicity.parquet")
    df.to_json(
        data_dir / "thoroughly_engineered_toxicity.jsonl", orient="records", lines=True
    )
    print("Saved", data_dir / "thoroughly_engineered_toxicity.jsonl")

    # RTP
    url = "https://raw.githubusercontent.com/alisawuffles/DExperts/refs/heads/main/generations/toy_prompt/gpt2/prompted_gens_gpt2.jsonl"
    download_file(url, data_dir / "prompted_gens_gpt2.jsonl")

    # Jigsaw
    (data_dir / "jigsaw").mkdir(exist_ok=True, parents=True)
    url = "https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge/resolve/main/train.csv"
    download_file(url, data_dir / "jigsaw/train.csv")
    url = "https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge/resolve/main/test.csv"
    download_file(url, data_dir / "jigsaw/test.csv")
    url = "https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge/resolve/main/test_labels.csv"
    download_file(url, data_dir / "jigsaw/test_labels.csv")

    # Coco captions
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    download_and_extract_zip(url, data_dir / "coco_captions_2017")
    os.system(
        f"mv {data_dir / 'coco_captions_2017/annotations/*'} {data_dir / 'coco_captions_2017'}"
    )
    os.system(f"rm -rf {data_dir / 'coco_captions_2017/annotations'}")
