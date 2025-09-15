import sys
import json
import requests as req
from openai import OpenAI
from tqdm import tqdm
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

"""
This is a script for generating multi-turn conversations about the visual characteristics of herbarium specimens. 
The script uses GPT-4 to generate conversations between a user and an AI assistant. 
The conversations are based on the visual characteristics of the specimens, as described in the provided descriptions.
"""

client = OpenAI()

species_found = json.load(open("species_captions_descriptions.json"))

# Duplicate each species entry to generate multiple conversations for each species
species_found = species_found * 5

system_prompt = """
You are an AI assistant to a herbarium scientist. 

You are provided with a species name of a herbarium specimen. You are also provided with a description of the species's visual characteristics from Wikipedia.
Unfortunately, you don't have access to the actual image.

Your task is to generate a multi-turn conversation with the user about the species's visual characteristics, for the purposes of VQA. 

Below are requirements for generating the questions and answers in the conversation:
- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or
names, as these may reveal the conversation is based on the text information, rather than
the image itself. Focus on the visual aspects of the image that can be inferred without
the text information.
- Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead,
refer to the information as being "in the image."
- Ensure that questions are diverse and cover a range of visual aspects of the image.
- Ensure that the questions are complex, going beyond just a visual summary - they should be rigorous and require careful understanding of the image.
- The conversation should include at least 2-3 turns of questions and answers about the
visual aspects of the image.
- Answer responsibly, avoiding overconfidence, and do not provide botanical advice or
diagnostic information. Encourage the user to consult a botanist for advice.

Generate the conversation in a JSON-like format, with the following structure:

{
"from": "human",
"value": "..."
},
{
"from": "assistant",
"value": "..."
}
"""

dataset = []

for sp in tqdm(species_found, desc="Generating conversations"):
    d = {}
    conversations = []

    user_prompt = f"""
    Species: {sp['species']}
    Descriptions: {sp['description']}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.8
    )

    content = response.choices[0].message.content.replace(
        "```json", "").replace("```", "").strip()

    content = "[" + content + "]"

    turns = json.loads(content)

    d["id"] = str(uuid.uuid4())
    d["species"] = sp["species"]
    d['caption'] = sp['caption']
    d["conversations"] = turns

    dataset.append(d)

json.dump(dataset, open("./descriptions-dataset.json", "w"))
