import os

from typing_extensions import Literal


def get_vulnerable_ckpt(model_name: Literal["attVgg16", "passt"],
                        dataset_name: Literal["asv19", "asv19-silence", "timi"]):
    """
    Returns vulnerable (i.e. not trained on attacked samples) checkpoint path of given dataset and model, from .env variables
    """
    ckpt_path = os.getenv("ATT-VGG-16-ASV19-CKP-DIR" if model_name == "attVgg16" else "PASST-TIME-SHIFT-ASV19-CKP-DIR")
    if dataset_name == "timi":
        ckpt_path = os.getenv("ATT-VGG-16-TIMI-CLEAN-CKP-DIR") if model_name == "attVgg16" else os.getenv(
            "PASST-TIMI-CLEAN-CKP-DIR")
    return ckpt_path


def get_fixed_ckpt():
    ckpt_path = os.getenv("PASST-ATTACK-AUG-ASV19-CKP-DIR_V3")
    return ckpt_path
