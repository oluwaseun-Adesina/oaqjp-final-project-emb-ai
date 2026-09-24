"""Emotion detection using the Watson NLP EmotionPredict function."""
import json
import requests

URL = ('https://sn-watson-emotion.labs.skills.network/v1/watson.runtime'
       '.nlp.v1/NlpService/EmotionPredict')
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def emotion_detector(text_to_analyze):
    """Detect the emotions expressed in the given text.

    Sends the text to the Watson NLP EmotionPredict function and extracts
    the anger, disgust, fear, joy and sadness scores, along with the
    dominant emotion (the one with the highest score).

    Args:
        text_to_analyze: the text to analyze.

    Returns:
        A dict with keys 'anger', 'disgust', 'fear', 'joy', 'sadness' and
        'dominant_emotion'. All values are None when the input text is
        blank (the API responds with status code 400 in that case).
    """
    input_json = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, json=input_json, headers=HEADERS)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    emotions = json.loads(response.text)['emotionPredictions'][0]['emotion']

    scores = {
        'anger': emotions['anger'],
        'disgust': emotions['disgust'],
        'fear': emotions['fear'],
        'joy': emotions['joy'],
        'sadness': emotions['sadness']
    }
    scores['dominant_emotion'] = max(scores, key=scores.get)

    return scores
