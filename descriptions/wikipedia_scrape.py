import re
import requests as req
from tqdm import tqdm
import pandas as pd
import json
from constants import KAGGLE_HERBARIUM_22_TRAIN_CSV

df = pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV)
print(df.columns)

output_file = "wiki_descriptions.json"

labels = (df["genus"] + " " + df["species"]).unique().tolist()
print(f"Number of unique species (labels): {len(labels)}")
print(f"Labels: {labels[:5]}")

species_found = []
n_found = 0
for sp in tqdm(labels):
    page = sp.replace(" ", "_")
    URL = f"https://en.wikipedia.org/w/index.php?title={page}&action=raw"
    print(URL)
    break
    response = req.get(URL)
    if "Wikimedia Error" not in response.text:
        remove_curly = re.sub(r"{{.*?}}", "", response.text, flags=re.DOTALL)
        remove_html = re.sub(r"<.*?>", "", remove_curly, flags=re.DOTALL)
        remove_brackets = re.sub(
            r"\[\[Category.*?\]\]", "", remove_html, flags=re.DOTALL)
        desc = remove_brackets.strip()

        tag = "== Description =="
        start = desc.find(tag)
        end = desc.find("==", start + len(tag))

        if start != -1 and end != -1:
            relevant_text = desc[start:end].strip()
            species_found.append({"species": sp, "description": relevant_text})
        else:
            species_found.append({"species": sp, "description": desc})

        n_found += 1

    if n_found % 100 == 0 and n_found > 0:
        print(f"Number of species descriptions found: {n_found}")

print(f"Number of species descriptions found: {len(species_found)}")


json.dump(species_found, open("species_found.json", "w"))

first = species_found[0]
print(f"Species: {first['species']}")
print(f"Description: {first['description'][:200]}")
print(f"Length: {len(first['description'])}")
