import os

from typing_extensions import Literal

from constants.env_var_enum import EnvVarEnum


def get_vulnerable_ckpt(model_name: Literal["attVgg16", "passt"],
                        dataset_name: Literal["asv19", "asv19-silence", "timi"]):
    """
    Returns vulnerable (i.e. not trained on attacked samples) checkpoint path of given dataset and model, from .env variables
    """
    ckpt_path = os.getenv(
        EnvVarEnum.ATT_VGG16_ASV19_CKP.value if model_name == "attVgg16" else EnvVarEnum.PASST_ASV19_CKP.value)
    if dataset_name == "timi":
        ckpt_path = os.getenv(EnvVarEnum.ATT_VGG16_TIMIT_CLEAN_CKP.value) if model_name == "attVgg16" else os.getenv(
            EnvVarEnum.PASST_TIMIT_CLEAN_CKP.value)
    return ckpt_path


def get_fixed_ckpt():
    ckpt_path = os.getenv(EnvVarEnum.PASST_ROBUST_ASV19_CKP_DIR.value)
    return ckpt_path
