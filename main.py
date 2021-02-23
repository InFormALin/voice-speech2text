from speech_to_text import VoiceToText
import logging

# Set LogLevel ..
logging.basicConfig(level=20)


def main():
    """For Testing Purposes .."""
    path_to_model = "./models/deepspeech-0.9.3-models.pbmm"
    path_to_scorer = "./models/deepspeech-0.9.3-models.scorer"

    callback = lambda t: print(t)

    vtt = VoiceToText(path_to_model, path_to_scorer, callback, False)
    vtt.start()


if __name__ == '__main__':
    main()
