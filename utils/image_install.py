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

CWD = os.getcwd()

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{CWD}/image_install.log', level=logging.INFO, filemode='w')

# not_installed = open("not_installed.txt", "w")

INSTALL_PATH = "/projectnb/herbdl/data/harvard-herbaria/images"
GBIF_MULTIMEDIA_DATA = "/projectnb/herbdl/data/harvard-herbaria/gbif/multimedia.txt"

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
adapter = HTTPAdapter()  #max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

min_delay = 10
max_delay = 30

def download_image(gbif_id, image_url, local_path):
    image_response = session.get(image_url, stream=True)

    with open(local_path, 'wb') as out_file:
        shutil.copyfileobj(image_response.raw, out_file)
        
    del image_response


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--country", dest="country",
                        help="Country to download samples from", metavar="COUNTRY CODE")

    args = parser.parse_args()

    country = args.country

    df = pd.read_csv(GBIF_MULTIMEDIA_DATA, delimiter="\t", usecols=['gbifID', 'identifier'], on_bad_lines='skip')

    if country:
        df = df[df['countryCode'] == country]
    
    for index, row in df.iterrows():
        
        gbif_id = row['gbifID']
        image_url = row['identifier']

        # resp = req.get(f"https://api.gbif.org/v1/occurrence/{gbif_id}").json()
        
        image_host = req.utils.urlparse(image_url).hostname
        
        local_path = os.path.join(INSTALL_PATH, f"{gbif_id}.jpg")

        logger.info(f"Downloading {gbif_id} to {local_path}")
        logger.info(f"Image URL: {image_url}")

        try: 

            headers = {
                "User-Agent": random.choice(user_agents),
                "Connection": "keep-alive",
                "Referer": "https://scc-ondemand1.bu.edu/"
            }

            if os.path.exists(local_path):
                # logger.warn(f"Image {gbif_id} already exists")
                continue
            else:
                download_image(gbif_id, image_url, local_path)

                # time.sleep(random.uniform(min_delay, max_delay))
                logger.info(f"Downloaded {gbif_id} to {local_path}")


        except Exception as e:
            logger.error(f"Error downloading {gbif_id}: {e}")
            # not_installed.write(f"{gbif_id}\n")
            
        logger.info("-------------------------------------------------")

    not_installed.close()
