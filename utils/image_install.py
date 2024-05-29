import pandas as pd
import os
import shutil
import requests as req
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(filename='image_install.log', level=logging.INFO)

HARVARD_HERBARIA_DATA = "/projectnb/herbdl/data/harvard-herbaria/data.csv"
INSTALL_PATH = "/projectnb/herbdl/data/harvard-herbaria/images"

df = pd.read_csv(HARVARD_HERBARIA_DATA, delimiter="\t", usecols=['gbifID'])

for i in range(len(df)):
    
    gbif_id = df.iloc[i]['gbifID']

    resp = req.get(f"https://api.gbif.org/v1/occurrence/{gbif_id}").json()
    image_url = resp['media'][0]['identifier']
   
    local_path = os.path.join(INSTALL_PATH, f"{gbif_id}.jpg")

    logger.info(f"Downloading {gbif_id} to {local_path}")
    logger.info(f"Image URL: {image_url}")

    try: 
        if os.path.exists(local_path):
            logger.info(f"Image {gbif_id} already exists")
            continue

        image_response = req.get(image_url, stream=True)

        with open(local_path, 'wb') as out_file:
            shutil.copyfileobj(image_response.raw, out_file)
            
        del image_response
    except Exception as e:
        logger.error(f"Error downloading {gbif_id}: {e}")
        logger.info("-------------------------------------------------")
        continue

    time.sleep(10)
    logger.info(f"Downloaded {gbif_id} to {local_path}")
    logger.info("-------------------------------------------------")
