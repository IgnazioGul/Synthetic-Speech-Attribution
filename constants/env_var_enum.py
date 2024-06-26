from enum import Enum


class EnvVarEnum(Enum):
    TIMIT_TTS_ROOT_DIR = "TIMIT_TTS_ROOT_DIR"
    ASV19_ROOT_DIR = "ASV19_ROOT_DIR"
    ASV19_SILENCE_ROOT_DIR = "ASV19_SILENCE_ROOT_DIR"

    ATT_VGG16_ASV19_CKP = "ATT_VGG16_ASV19_CKP"
    ATT_VGG16_TIMIT_CLEAN_CKP = "ATT_VGG16_TIMIT_CLEAN_CKP"

    PASST_ASV19_CKP = "PASST_ASV19_CKP"
    PASST_TIMIT_CLEAN_CKP = "PASST_TIMIT_CLEAN_CKP"
    PASST_ROBUST_ASV19_CKP_DIR = "PASST_ROBUST_ASV19_CKP_DIR"

    PASST_ASV19_SILENCE_CKP = "PASST_ASV19_SILENCE_CKP"
    ATT_VGG16_ASV19_SILENCE_CKP_DIR = "ATT_VGG16_ASV19_SILENCE_CKP_DIR"
