import pandas as pd
import os
import shutil
import logging
import time
import random
from image_utils import get_file_size_in_mb, resize_with_aspect_ratio
from concurrent.futures import ThreadPoolExecutor, as_completed

CWD = os.getcwd()

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{CWD}/image_resize.log', level=logging.INFO, filemode='w')

INSTALL_PATH = "/projectnb/herbdl/data/harvard-herbaria/images"

N_RESIZED = 0

def process_image(image_path):
    global N_RESIZED
    i = os.path.basename(image_path)
    
    orig_size = get_file_size_in_mb(image_path)

    if orig_size == 0:
        logger.warning(f"File {i} has size 0. Skipping...")
        return
    elif orig_size < 2:
        logger.warning(f"File {i} is less than 2 MB. Skipping...")
        return

    logger.info(f"Resizing {i} from {orig_size:.2f} MB...")
    result = resize_with_aspect_ratio(image_path, image_path, logger)
    if result:
        logger.info(f"Resized {i} to {get_file_size_in_mb(image_path):.2f} MB")
        N_RESIZED += 1

    if N_RESIZED % 20000 == 0:
        logger.info(f"Resized {N_RESIZED} images")


if __name__ == "__main__":
    images = [os.path.join(INSTALL_PATH, i) for i in os.listdir(INSTALL_PATH)]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_image, image) for image in images]
        for future in as_completed(futures):
            future.result()
