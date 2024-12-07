"""
Author: Jakub Lisowski, 2024
MIT License
"""

import os
import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import py7zr
import requests

SCRIPT_PATH = str(Path.resolve(Path(f'{__file__}/../')))
TMP_PATH = str(os.path.join(SCRIPT_PATH, '../.tmp'))
PGN_LINK = 'https://www.pgnmentor.com/openings/KIDOther7.zip'
EXTRACTION_DIR = 'extracted'
PGN_PATH = str(os.path.join(TMP_PATH, EXTRACTION_DIR))
CHECKMATE_CHARIOT_DATA_PATH = str(os.path.join(SCRIPT_PATH, '../src/ported/engine/Checkmate-Chariot/Tests/'))
CHECKMATE_CHARIOT_ROOK_PATH = str(os.path.join(CHECKMATE_CHARIOT_DATA_PATH, 'RookTests.tar.7z'))
CHECKMATE_CHARIOT_BISHOP_PATH = str(os.path.join(CHECKMATE_CHARIOT_DATA_PATH, 'BishopTests.tar.7z'))
CHECKMATE_CHARIOT_UTILS_DB_GEN_PATH = str(
    os.path.join(SCRIPT_PATH, '../utils/scripts/random_fen_position_db_create.py'))
DATA_PATH_OUTPUT = str(os.path.join(SCRIPT_PATH, '../src/tests/test_data'))

FEN_DB_COMMAND_ARGS = ["--pgn-file", f"{PGN_PATH}/KIDOther7.pgn",
                       "--out-path", f"{DATA_PATH_OUTPUT}/fen_db.txt",
                       "--min-moves", "5",
                       "--max-moves", "15",
                       "--output-size", "100000"]


def download_and_extract_zip(url: str, download_dir: str) -> None:
    filename = url.split('/')[-1]
    filepath = os.path.join(download_dir, filename)

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Download completed: {filepath}")

    extract_path = os.path.join(download_dir, EXTRACTION_DIR)
    print(f"Extracting to {extract_path}...")

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction completed: {extract_path}")


def extract_7z_tar_file(path: str, output_dir: str) -> None:
    print(f"Extracting {path}...")
    with py7zr.SevenZipFile(path, mode='r') as z:
        extracted = z.readall()

        tar_filename = list(extracted.keys())[0]
        tar_data = extracted[tar_filename]
        tar_bytes = BytesIO(tar_data.read() if hasattr(tar_data, 'read') else tar_data)

        with tarfile.open(fileobj=tar_bytes, mode='r:*') as tar:
            os.makedirs(output_dir, exist_ok=True)
            tar.extractall(path=output_dir)

    print(f"Successfully extracted {path} to {output_dir}")


def extract_positions() -> None:
    archives = [
        CHECKMATE_CHARIOT_ROOK_PATH,
        CHECKMATE_CHARIOT_BISHOP_PATH
    ]

    for archive in archives:
        extract_7z_tar_file(archive, DATA_PATH_OUTPUT)


def generate_fen_db() -> None:
    print(f"Generating FEN database using {CHECKMATE_CHARIOT_UTILS_DB_GEN_PATH}")

    python_executable = sys.executable
    command = [python_executable, CHECKMATE_CHARIOT_UTILS_DB_GEN_PATH]
    command.extend(FEN_DB_COMMAND_ARGS)

    subprocess.run(command,
                   check=True,
                   text=True)

    print("FEN database generation completed successfully")


def main() -> None:
    try:
        os.makedirs(TMP_PATH, exist_ok=True)
        download_and_extract_zip(PGN_LINK, TMP_PATH)
        extract_positions()
        generate_fen_db()
    except Exception as e:
        print(f"Failed due to error: {e}")
        return


if __name__ == '__main__':
    main()
