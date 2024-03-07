TRACKS_MAX_DURATION_SEC = 9.600

classes_names = ["gtts", "tacotron", "glowtts", "fastpitch", "vits", "fastspeech2", "mixertts", "mixertts_x",
                 "speedyspeech", "tacotron2", "talknet", "silero"]
LABELS_MAP = {val: idx for idx, val in enumerate(classes_names)}
AUDIO_KEY = "audio"
CLASS_KEY = "class"
ORIGINAL_SPEC_KEY = "original_spec"

CSV_METADATA_HEADER = [AUDIO_KEY, CLASS_KEY]
