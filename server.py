"""Flask server exposing the emotion detector as `/emotionDetector`.

The server expects a form field or JSON body with key `text`.
"""
from __future__ import annotations

from typing import Dict, Any

from flask import Flask, request, render_template_string

from EmotionDetection import emotion_detector


app = Flask(__name__)


INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Emotion Detector</title>
  </head>
  <body>
    <h1>Emotion Detector</h1>
    <form action="/emotionDetector" method="post">
      <label for="text">Enter text:</label><br>
      <input type="text" id="text" name="text" size="80"><br><br>
      <input type="submit" value="Analyze">
    </form>
    {% if message %}
      <p style="font-family: Courier New; color: DodgerBlue;">{{ message|safe }}</p>
    {% endif %}
  </body>
</html>"""


def _format_message(result: Dict[str, Any]) -> str:
    """Format the detector result into the string required by the lab."""
    if not result or result.get("dominant_emotion") is None:
        return "Invalid text! Please try again!"

    return (
        "For the given statement, the system response is '") + \
        f"anger: {result['anger']}, 'disgust': {result['disgust']}, 'fear': {result['fear']}, 'joy': {result['joy']} and 'sadness': {result['sadness']}. The dominant emotion is <b>{result['dominant_emotion']}</b>."


@app.route("/", methods=["GET"])
def index():
    """Render a simple index page with a form."""
    return render_template_string(INDEX_HTML, message=None)


@app.route("/emotionDetector", methods=["POST"])
def emotion_detector_route():
    """Handle form or JSON requests and display formatted output."""
    text = request.form.get("text") or request.json and request.json.get("text")
    result = emotion_detector(text)
    message = _format_message(result)
    return render_template_string(INDEX_HTML, message=message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
