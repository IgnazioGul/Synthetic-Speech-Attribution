from enum import Enum


class DatasetEnum(Enum):
    TIMIT_TTS = "timit_tts"
    ASV19 = "asv19"
    ASV19_SILENCE = "asv19_silence"


dataset_classes_map = {
    DatasetEnum.TIMIT_TTS.value: 12,
    DatasetEnum.ASV19.value: 7,
    DatasetEnum.ASV19_SILENCE.value: 7
}
