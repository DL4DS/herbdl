import re
from tqdm import tqdm
import pandas as pd
import json
from constants import KAGGLE_HERBARIUM_22_TRAIN_CSV
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from dotenv import load_dotenv

load_dotenv()
"""
This script scrapes descriptions of species from Wikipedia, cleans them using GPT-4, creates simpler captions and saves all to a JSON file.
"""

df = pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV)

# Take top N species by frequency
N = 100
df = df.groupby(["genus", "species"]).size().reset_index(name="count")
df = df.sort_values(by="count", ascending=False)
df = df.head(N)

labels = (df["genus"] + " " + df["species"]).unique().tolist()

# Set your OpenAI API key
client = OpenAI()

# Wikipedia scraping function
def scrape_wikipedia(sp):
    """Scrapes Wikipedia raw page text for a given species."""
    page = sp.replace(" ", "_")
    url = f"https://en.wikipedia.org/w/index.php?title={page}&action=raw"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200 or "Wikimedia Error" in response.text:
            return None  # Skip if the page is not found

        # Clean extracted text
        text = response.text
        text = re.sub(r"{{.*?}}", "", text, flags=re.DOTALL)  # Remove infoboxes
        text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)  # Remove HTML tags
        text = re.sub(r"\[\[Category.*?\]\]", "", text, flags=re.DOTALL)  # Remove categories

        tag = "== Description =="
        start = text.find(tag)
        end = text.find("==", start + len(tag))

        if start != -1 and end != -1:
            return {"species": sp, "description": text[start:end].strip()}
        else:
            return {"species": sp, "description": text.strip()}

    except Exception as e:
        return None

# Generate improved descriptions using GPT-4o
def improve_description(species, description):
    """Enhances the species description using GPT-4o."""
    prompt = f"""Improve the following plant description for {species}. Focus on its visual characteristics, origin, geographic distribution, and unique traits.
    
    Original description:
    {description}

    Improved description:"""

    try:

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a botany expert."},
                      {"role": "user", "content": prompt}],
        temperature=0.7
    )

        content = response.choices[0].message.content.replace(
        "```json", "").replace("```", "").strip()
        return content

    except Exception as e:
        print(f"Failed to enhance description for {species}: {str(e)}")
        return description  # Fallback to original if GPT call fails


def create_caption(species, description):
    """Enhances the species description using GPT-4o."""
    prompt = f"""Provide a very short, high-level image caption for the species {species}. You don't have access to the image. Make your caption of the following format:
    Species name: <species_name> | Common Name: <common_name> | Leaves/Petals: <color, shape, texture> | Habitat: <where it is typically found>
    
    Description:
    {description}

    Caption:"""

    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a botany expert."},
                      {"role": "user", "content": prompt}],
        temperature=0.5
    )
        content = response.choices[0].message.content.replace(
        "```json", "").replace("```", "").strip()
        return content

    except Exception as e:
        print(f"Failed to create caption for {species}: {str(e)}")
        return description  # Fallback to original if GPT call fails


def improve_description_local(species, description):
    model = LlavaNextForConditionalGeneration.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=".")
    prompt = f"""Improve the following plant description for {species}. Focus on its visual characteristics, origin, geographic distribution, and unique traits.

    Original description:
    {description}

    Improved description:"""
    processor = LlavaNextProcessor.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=".")

    inputs = processor(prompt, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=300)
    response = processor.decode(output[0], skip_special_tokens=True)

    return response

# Step 1: Scrape Wikipedia pages in parallel
species_found = []
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust thread count as needed
    futures = {executor.submit(scrape_wikipedia, sp): sp for sp in labels}
    for future in tqdm(as_completed(futures), total=len(labels), desc="Scraping Wikipedia"):
        result = future.result()
        if result:
            species_found.append(result)

print(f"Number of species descriptions found: {len(species_found)}")

# Step 2: Create captions and improve descriptions in parallel
captions_and_descriptions = []

with ThreadPoolExecutor(max_workers=5) as executor:
    # Generate captions
    captions_futures = {executor.submit(create_caption, entry["species"], entry["description"]): entry for entry in species_found}
    captions_results = {}
    for future in tqdm(as_completed(captions_futures), total=len(species_found), desc="Creating captions"):
        result = future.result()
        if result:
            species = captions_futures[future]["species"]
            captions_results[species] = result  # Store captions by species

    # Generate enhanced descriptions
    desc_futures = {executor.submit(improve_description, entry["species"], entry["description"]): entry for entry in species_found}
    for future in tqdm(as_completed(desc_futures), total=len(species_found), desc="Enhancing descriptions"):
        result = future.result()
        if result:
            species = desc_futures[future]["species"]
            captions_and_descriptions.append({
                "species": species,
                "caption": captions_results.get(species, ""),  # Retrieve corresponding caption
                "description": result
            })

# Save combined captions & descriptions into a single JSON file
output_file = "species_captions_descriptions.json"
with open(output_file, "w") as f:
    json.dump(captions_and_descriptions, f, indent=4)

print(f"Captions and descriptions saved to {output_file}")
