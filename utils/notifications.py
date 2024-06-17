import requests
import json
import os

URL = "https://api.pushover.net/1/messages.json"

def send_notification(title, message):
    data = {
        "token": os.getenv("PUSHOVER_API_TOKEN"),
        "user": os.getenv("PUSHOVER_USER_KEY"),
        "title": title,
        "message": message
    }
    requests.post(URL, data=data)


# send_notification("Test", "This is a test message")