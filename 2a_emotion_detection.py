"""Task 2: emotion detection application using the Watson NLP library."""
import requests

URL = ('https://sn-watson-emotion.labs.skills.network/v1/watson.runtime'
       '.nlp.v1/NlpService/EmotionPredict')
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def emotion_detector(text_to_analyze):
    """Send text to the Watson NLP EmotionPredict function and return the
    raw response text."""
    input_json = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, json=input_json, headers=HEADERS)
    return response.text
