#!/usr/bin/python3
import os
import glob
import kaggle
import argparse


def fetch_kaggle_data(dataset: str, out_path: str, out_name: str) -> str:
    """
    Download and rename a Kaggle dataset.
    :param dataset: Name of Kaggle dataset to download.
    :param out_path: Location where dataset will be stored.
    :param out_name: User-specified name of the downloaded dataset.
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=out_path, unzip=True)
    files = glob.glob(f"{out_path}/*")
    newest_file = max(files, key=os.path.getmtime)
    os.rename(f"{newest_file}", f"{out_path}/{out_name}")
    return 'File download and renaming has been completed.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of the Kaggle dataset to download.")
    parser.add_argument("out_path", type=str,
                        help="Path to where to download the dataset.")
    parser.add_argument("out_name", type=str,
                        help="Specify name of downloaded dataset..")
    args = parser.parse_args()

    fetch_kaggle_data(args.dataset, args.out_path, args.out_name)


if __name__ == "__main__":
    main()
