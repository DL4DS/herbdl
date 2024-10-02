import pandas as pd
import os
import shutil
import requests as req
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import time
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from notifications import send_notification
from image_utils import get_file_size_in_mb, resize_with_aspect_ratio
from PIL import UnidentifiedImageError


# get JOB_ID from environment
CWD = os.getcwd()

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{CWD}/image_install.log', level=logging.INFO, filemode='w')

link_logger = logging.getLogger("link_logger")
link_logger.setLevel(logging.INFO)

INSTALL_PATH = "/projectnb/herbdl/data/GBIF/IMAGES/"
GBIF_MULTIMEDIA_DATA = "/projectnb/herbdl/data/GBIF/multimedia.txt"

n_installed = len(os.listdir(INSTALL_PATH))

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
]

session = req.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

min_delay = 10
max_delay = 30

def download_image(gbif_id, image_url, local_path):
    try:
        image_response = session.get(image_url, stream=True, headers={
            "User-Agent": random.choice(user_agents),
            "Connection": "keep-alive",
            "Referer": "https://scc-ondemand1.bu.edu/"
        })

        if image_response.status_code == 200:

            if image_response.headers.get('Content-Type') != 'image/jpeg':
                logger.error(f"Invalid content type for {gbif_id}: {image_response.headers.get('Content-Type')}. Skipping. ")
                link_logger.info(f"{gbif_id}\t{image_url}")
                del image_response
                return False

            with open(local_path, 'wb') as out_file:
                shutil.copyfileobj(image_response.raw, out_file)

            logger.info(f"Downloaded {gbif_id} to {local_path}")
            del image_response

            return True
        else:
            raise Exception(f"HTTP {image_response.status_code}")

    except Exception as e:
        logger.error(f"Error downloading {gbif_id}: {e}")
        return False


def resize_image(gbif_id, local_path):
    result = resize_with_aspect_ratio(local_path, local_path)
    if result:
        logger.info(f"Resized {gbif_id} to 1000x1000. Path: {local_path}")
    else:
        logger.info(f"Skipped resizing {gbif_id} as it is already 1000x1000")
    

def process_row(row):
    global n_installed
    gbif_id = row['gbifID']
    image_url = row['identifier']
    local_path = os.path.join(INSTALL_PATH, f"{gbif_id}.jpg")

    downloaded=False

    if os.path.exists(local_path):
        #logger.warning(f"Image {gbif_id} already exists")
        size = get_file_size_in_mb(local_path)
        if size < 0.01:
            logger.warning(f"Image {gbif_id} is too small ({size} MB), redownloading")
            downloaded = download_image(gbif_id, image_url, local_path)
    else:
        logger.info(f"Downloading {gbif_id} to {local_path}. Image URL: {image_url}")
        downloaded = download_image(gbif_id, image_url, local_path)

    if downloaded:
        n_installed += 1

    if n_installed % 50000 == 0:
        # send_notification("Image Installation", f"Installed {n_installed} images")
        logger.info(f"Installed {n_installed} images")

    try:
        if downloaded:
            resize_image(gbif_id, local_path)

    except (OSError, UnidentifiedImageError) as e:            
        os.remove(local_path)
        logger.error(f"Error resizing {gbif_id}: {e}.")
        # process_row(row)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--country", dest="country",
                        help="Country to download samples from", metavar="COUNTRY CODE")

    args = parser.parse_args()
    country = args.country

    df = pd.read_csv(GBIF_MULTIMEDIA_DATA, delimiter="\t", usecols=['gbifID', 'identifier'], on_bad_lines='skip')

    if country:
        df = df[df['countryCode'] == country]

    send_notification("Image Installation", f"Starting image installation for {len(df)} images")

    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(process_row, row) for index, row in df.iterrows()]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")