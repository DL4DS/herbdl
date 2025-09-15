import requests
from tqdm import tqdm
import pandas as pd
from constants import KAGGLE_HERBARIUM_22_TRAIN_CSV
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV)
labels = (df["genus"] + " " + df["species"]).unique().tolist()

URL = "https://plants.ces.ncsu.edu/plants/"

total = len(labels)
scraped_species = []

def scrape_species(sp):
    species = sp.lower().replace(" ", "-")
    response = requests.get(URL + species)
    if response.status_code == 200:
        return sp  # Return the original species name if found
    return None  # Return None if not found


total = len(labels)

with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
    futures = {executor.submit(scrape_species, sp): sp for sp in labels}

    for future in tqdm(as_completed(futures), total=total, desc="Scraping species"):
        result = future.result()
        if result:
            scraped_species.append(result)

# Write successfully scraped species to a file
output_file = "scraped_species.txt"
with open(output_file, "w") as f:
    for species in scraped_species:
        f.write(species + "\n")

print(f"Scraped {len(scraped_species)} out of {total} species.")
print(f"Successfully scraped species saved to {output_file}")
