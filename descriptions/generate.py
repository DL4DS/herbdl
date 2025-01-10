import sys
import json
import requests as req
from openai import OpenAI
from tqdm import tqdm
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

species_found = json.load(open("species_found.json"))

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
- Ensure that the questions are not too easy for a herbarium scientist to answer - they should be modeled after the questions asked by herbarium scientists.
- The conversation should include at least 2-3 turns of questions and answers about the
visual aspects of the image.
- Answer responsibly, avoiding overconfidence, and do not provide botanical advice or
diagnostic information. Encourage the user to consult a botanist for advice.

Generate the conversation in a JSON-like format, with the following structure:

{
"from": "human",
"text": "..."
},
{
"from": "assistant",
"text": "..."
}
"""

dataset = []

for sp in list(species_found.keys()):
    d = {}
    conversations = []

    user_prompt = f"""
    Species: {sp}
    Descriptions: {species_found[sp]}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6
    )

    content = response.choices[0].message.content.replace(
        "```json", "").replace("```", "").strip()

    content = "[" + content + "]"

    turns = json.loads(content)

    d["id"] = str(uuid.uuid4())
    d["species"] = sp
    d["conversations"] = turns

    dataset.append(d)

json.dump(dataset, open("descriptions-dataset.json", "w"))
