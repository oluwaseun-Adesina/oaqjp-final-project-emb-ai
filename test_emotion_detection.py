"""Unit tests for the EmotionDetection package.

These tests use the environment variable `EMOTION_DETECTOR_FORCE_LOCAL`
to ensure the local deterministic analyzer is used so tests are stable
even without network access.
"""
import os

from EmotionDetection import emotion_detector


def setup_module():
    os.environ["EMOTION_DETECTOR_FORCE_LOCAL"] = "1"


def teardown_module():
    os.environ.pop("EMOTION_DETECTOR_FORCE_LOCAL", None)


def assert_dominant(text: str, expected: str) -> None:
    res = emotion_detector(text)
    assert res["dominant_emotion"] == expected


def test_joy():
    assert_dominant("I am glad this happened", "joy")


def test_anger():
    assert_dominant("I am really mad about this", "anger")


def test_disgust():
    assert_dominant("I feel disgusted just hearing about this", "disgust")


def test_sadness():
    assert_dominant("I am so sad about this", "sadness")


def test_fear():
    assert_dominant("I am really afraid that this will happen", "fear")
