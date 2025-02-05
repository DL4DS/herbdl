import requests
import json
from tqdm import tqdm
import sys

URL = "https://plants.ces.ncsu.edu/plants/"
species_found = json.load(open("species_found.json"))

print(f"Number of species found: {len(species_found)}")

total = len(species_found)
scraped = 0

for sp in tqdm(species_found, desc="Scraping species"):
    species = sp["species"].lower().replace(" ", "-")
    response = requests.get(URL + species)
    if response.status_code == 200:
        scraped += 1


print(f"Scraped {scraped} out of {total} species")
