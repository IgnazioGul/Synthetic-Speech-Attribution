# Synthetic-Speech-Attribution

This project is the implementation of the research work "Explainable detection of deepfake audio generators".
The main objective of the research is to study the process of deepfake audio recognition in order to highlight the "
fingerprint" (i.e. the most characteristic features) that each synthetic speech generator leaves in his samples.
This analysis is carried out through the analysis of the decision process of the models through SHAP, while each
hypothesis on the most characteristic features present in the samples is verified through the use of adversarial attacks
that have the objective to obscure those features.

## Setup

### Environment variables

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

### Install dependencies

To install the dependencies, create a virtual environment with the following command:

```bash
python -m venv venv 
```

Then, activate the virtual environment:

```bash
source venv/bin/activate
```

Finally, install the dependencies:

```bash
pip install -r requirements.txt
```

### Preprocess TIMIT-TTS labels

Run the following command to preprocess the TIMIT-TTS labels:

```bash
python ./utils/timit_tts_utils.py
```

## Usage

### Training for the SSA task

The `train.py` script contains all the configuration of the training, so read it carefully and modify it according to
your needs.
Then, run the following command from the root folder of the project:

```bash
python train.py
```

### Testing the attacks

The `./attacks/test_attack.py` script contains all the configuration of the attack, so read it carefully and modify it
according to your needs.
Then, run the following command from the root folder of the project:

```bash
python ./attacks/test_attack.py
```

## This projects is in WIP stage, further informations will be provided in future.
