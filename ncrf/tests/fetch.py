"""Test-data download and loading helpers for the NCRF regression suite."""

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
import os
import pickle
import pooch
import certifi
import operator
import logging

from pathlib import Path
from shutil import rmtree
from typing import Any


logger = logging.getLogger(__name__)

os.environ.setdefault('SSL_CERT_FILE', certifi.where())

RELEASE = '0.2'
TESTING_VERSIONED = f"NCRF-testing-data-{RELEASE}"

# web url to fetch the file
archive_name = f"NCRF-testing-data-{RELEASE}.tar.gz"
url = f"https://codeload.github.com/proloyd/NCRF-testing-data/tar.gz/{RELEASE}"
known_hash = "sha256:eb9449d0f34eef1a72599a212d10e9b2d5a2a00cc08743db675f4e9248e68f7e"
folder_name = "ncrf-testing-data"

# names for lading data
_NAMES = ('meg', 'stim', 'fwd_sol', 'emptyroom')


def fetch_dataset(force_download: bool = False) -> Path:
    """Ensure the versioned test dataset is available locally and return its path."""
    # manage local storage
    root_dir = Path(os.path.realpath(os.path.join(__file__, '..', '..', '..')))
    final_path = root_dir / folder_name

    want_version = RELEASE
    ver_fname = final_path / "version.txt"
    outdated = False
    if ver_fname.exists():
        with open(ver_fname, 'r') as fid:
            data_version = fid.readline().strip()
        outdated = operator.gt(want_version, data_version)
    else:
        logger.info(
            f"Dataset {folder_name} was not found, downloading"
            f"latest version {want_version}"
        )
        force_download = True

    if outdated:
        logger.info(
            f"Dataset {folder_name} version {data_version} out of date, "
            f"latest version is {want_version}"
        )
        force_download = True

    if not outdated and not force_download:
        return final_path

    # Prepare pooch to fetch the data
    processor = pooch.Untar(extract_dir=root_dir)
    downloader_params = dict(timeout=15, progressbar=True)
    downloader = pooch.HTTPDownloader(**downloader_params)

    urls = {archive_name: url}
    registry = {archive_name: known_hash}
    fetcher = pooch.create(
        path=str(root_dir),
        base_url="",  # Full URLs are given in the `urls` dict.
        version=None,  # Data versioning is decoupled from MNE-Python version.
        urls=urls,
        registry=registry,
        retry_if_failed=2,  # 2 retries = 3 total attempts
    )

    try:
        fetcher.fetch(
            fname=archive_name, downloader=downloader, processor=processor
        )
    except ValueError as err:
        err = str(err)
        if "hash of downloaded file" in str(err):
            raise ValueError(
                f"{err} Consider updating hash of the dataset"
            )
        else:
            raise
    fetcher.fetch(
        fname=archive_name, downloader=downloader, processor=processor
    )
    fname = root_dir / archive_name
    fname.unlink()

    rmtree(final_path, ignore_errors=True)
    os.replace(root_dir / TESTING_VERSIONED, final_path)
    return final_path


def load(name: str) -> Any:
    """Load one of the packaged regression-test fixtures by name."""
    folder_name = fetch_dataset()
    if name in _NAMES:
        fname = os.path.join(folder_name, f"{name}.pickled")
    else:
        raise ValueError(f"{name}: not found, check dataset version")
    with open(fname, 'rb') as f:
        v = pickle.load(f)
    return v
