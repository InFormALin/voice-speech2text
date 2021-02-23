import logging
import collections
import queue
import os.path
import deepspeech
import numpy as np
import pyaudio
import webrtcvad


class MicrophoneAudio:
    """This class streams the audio from the default microphone to a buffer ..
    The class is based on the example mic_vad_streamin.py of DeepSpeech Examples
    (see https://github.com/mozilla/DeepSpeech-examples/tree/master/mic_vad_streaming)
    This class combines the classes Audio and VADAudio """

    # Some basic constants ..
    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, aggressiveness=3):
        self._buffer_queue = queue.Queue()
        self._input_rate = MicrophoneAudio.RATE_PROCESS
        self._sample_rate = MicrophoneAudio.RATE_PROCESS
        self._block_size = int(MicrophoneAudio.RATE_PROCESS / float(MicrophoneAudio.BLOCKS_PER_SECOND))
        self._block_size_input = int(self._input_rate / float(MicrophoneAudio.BLOCKS_PER_SECOND))
        self._audio = pyaudio.PyAudio()
        self._vad = webrtcvad.Vad(aggressiveness)

        def callback(in_data, frame_count, time_info, status):
            self._buffer_queue.put(in_data)
            return None, pyaudio.paContinue

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self._input_rate,
            'input': True,
            'frames_per_buffer': self._block_size_input,
            'stream_callback': callback,
        }

        self.stream = self._audio.open(**kwargs)
        self.stream.start_stream()

    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()
        self._audio.terminate()

    def _frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        while True:
            yield self._buffer_queue.get()

    def vad_collector(self, padding_ms=300, ratio=0.75):
        """Generator that yields series of consecutive audio frames comprising each utterence,
        separated by yielding a single None. Determines voice activity by ratio of frames in padding_ms.
        Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        frames = self._frame_generator()
        num_padding_frames = padding_ms // (1000 * self._block_size // self._sample_rate)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self._vad.is_speech(frame, self._sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


class VoiceToText:
    """This class uses deepspeech to recognize voice
    :param path_to_model the relative or absolute path to the model
    :param path_to_scorer the relative or absolute path to the scorer
    :param callback a function that can consume the data from finishStream or finishStreamWithMetadata of a model stream
    :param callback_with_metadata indicator whether finishStream or finishStreamWithMetadata shall be called
    """
    def __init__(self, path_to_model, path_to_scorer, callback, callback_with_metadata=False):
        if not os.path.exists(path_to_model) or not os.path.exists(path_to_scorer):
            logging.error("Language models are not downloaded .. please download them!")
            raise ValueError("Model invalid")
        self._model = deepspeech.Model(model_path=path_to_model)
        self._model.enableExternalScorer(path_to_scorer)
        self._audio = MicrophoneAudio()
        self._callback = callback
        self._callback_with_metadata = callback_with_metadata

    def start(self):
        """Simply Start the Recognition"""
        frames = self._audio.vad_collector()
        stream = self._model.createStream()
        logging.info("Feeding Audio to Model")
        for frame in frames:
            if frame is not None:
                stream.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                logging.info("Recognizing Text")
                result = stream.finishStream() if not self._callback_with_metadata else stream.finishStreamWithMetadata()
                self._callback(result)
                stream = self._model.createStream()
                logging.info("Feeding Audio to Model")
