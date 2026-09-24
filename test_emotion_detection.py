"""Unit tests for the EmotionDetection package."""
import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test cases for the emotion_detector function."""

    def test_joy(self):
        """'I am glad this happened' should be dominated by joy."""
        result = emotion_detector('I am glad this happened')
        self.assertEqual(result['dominant_emotion'], 'joy')

    def test_anger(self):
        """'I am really mad about this' should be dominated by anger."""
        result = emotion_detector('I am really mad about this')
        self.assertEqual(result['dominant_emotion'], 'anger')

    def test_disgust(self):
        """'I feel disgusted just hearing about this' should be disgust."""
        result = emotion_detector('I feel disgusted just hearing about this')
        self.assertEqual(result['dominant_emotion'], 'disgust')

    def test_sadness(self):
        """'I am so sad about this' should be dominated by sadness."""
        result = emotion_detector('I am so sad about this')
        self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_fear(self):
        """'I am really afraid that this will happen' should be fear."""
        result = emotion_detector('I am really afraid that this will happen')
        self.assertEqual(result['dominant_emotion'], 'fear')


if __name__ == "__main__":
    unittest.main()
