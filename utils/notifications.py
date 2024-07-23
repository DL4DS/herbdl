import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def send_notification(title, message):

    URL = "https://api.pushover.net/1/messages.json"

    api_token = os.getenv("PUSHOVER_API_TOKEN")
    user_key = os.getenv("PUSHOVER_USER_KEY")

    if not api_token or not user_key:
        print("Pushover API token or user key not set")
        return

    data = {
        "token": os.getenv("PUSHOVER_API_TOKEN"),
        "user": os.getenv("PUSHOVER_USER_KEY"),
        "title": title,
        "message": message
    }
    requests.post(URL, data=data)
