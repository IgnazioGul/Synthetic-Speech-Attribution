# Synthetic-Speech-Attribution

This project builds up a Neural network approach on synthetic speech attribution.

A `.env` file must be placed in root-level directory, containing the following variables:

```
TIMIT_TTS_ROOT_DIR=""
ASV19_ROOT_DIR=""
ASV19_SILENCE_ROOT_DIR=""

ATT_VGG16_ASV19_CKP=""
ATT_VGG16_ASV19_SILENCE_CKP_DIR=""
ATT_VGG16_TIMIT_CLEAN_CKP=""

PASST_ASV19_CKP="" 
PASST_ASV19_SILENCE_CKP=""
PASST_TIMIT_CLEAN_CKP=""
PASST_ROBUST_ASV19_CKP_DIR="" 
```

> **Please note**: Each variable value must be enclosed in ""

It follows a description of each variable:

- `TIMIT_TTS_ROOT_DIR`: Root directory of the TIMIT-TTS dataset.
- `ASV19_ROOT_DIR`: Root directory of the ASVspoof2019 dataset.
- `ASV19_SILENCE_ROOT_DIR` (<span style="color:red">**Optional**</span>): Root directory of the ASVspoof2019 dataset,
  containing the silence samples.
- `ATT_VGG16_ASV19_CKP`: Full path of the checkpoint of the AttVgg16 model trained on the ASVspoof2019
  dataset.
- `ATT_VGG16_ASV19_SILENCE_CKP_DIR` (<span style="color:red">**Optional**</span>): Full path of the checkpoint of the
  AttVgg16 model trained on the
  silence-ASVspoof2019
  dataset.
- `ATT_VGG16_TIMIT_CLEAN_CKP`: Full path of the checkpoint of the AttVgg16 model trained on the TIMIT-TTS
  dataset.
- `PASST_ASV19_CKP`: Full path of the checkpoint of the PaSST model trained on the ASVspoof2019
  dataset.
- `PASST_ASV19_SILENCE_CKP` (<span style="color:red">**Optional**</span>): Full path of the checkpoint of the PaSST
  model trained on the
  silence-ASVspoof2019
  dataset.
- `PASST_TIMIT_CLEAN_CKP`: Full path of the checkpoint of the PaSST model trained on the TIMIT-TTS
  dataset.
- `PASST_ROBUST_ASV19_CKP_DIR` (<span style="color:red">**Optional**</span>): Full path of the checkpoint of the PaSST
  model trained on the **ASVspoof2019 dataset enriched with the attacked samples**.

## This projects is in WIP stage, further informations will be provided in future.
