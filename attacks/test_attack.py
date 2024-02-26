import os

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal

from dataset.load_asv_19_dataset import LoadAsvSpoof19
from dataset.load_timi_dataset import LoadTimiDataset
from synthetic_classifier import SyntheticClassifier
from utils.augmentation_viewer import print_specs
from utils.augmentations import (asv19_attack_class_based)
from utils.dataset_utils import extract_dict_vals, load_specific_classes, get_dataset_base_path
from utils.model_utils import get_vulnerable_ckpt, get_fixed_ckpt
from utils.timi_tts_constants import AUDIO_KEY


def load_txt(full_path):
    data = np.loadtxt(full_path, dtype=str, delimiter=' ')
    np.random.seed(123)
    np.random.shuffle(data)
    return data


def preprocess_audio(audio, sr, dataset_obj, should_return_spec=False):
    aug_audio, spec = dataset_obj.preprocess_item(audio, sr)
    if should_return_spec:
        return aug_audio.clone().detach().to(dtype=torch.float), spec
    else:
        return aug_audio.clone().detach().to(dtype=torch.float)


def get_model_kwargs(model_name="attVgg16", dataset="asv19", extract_manual_spec: bool = False, mode="normal",
                     pretrained: bool = True):
    isPretrained = pretrained
    optimizer = "Adam"
    lr = 0.000001
    decay = 0.00008
    batch_size = 32
    metadata_file = "aug_reduced.csv"
    # model_name = "resnet50"
    model_name = model_name
    isValidationEnabled = True
    is_augment_enabled = False
    dataset = dataset
    epochs = 25
    n_classes_timi = 2 if mode == "reduced" else 12
    n_classes = 7 if dataset == "asv19" or dataset == "asv19-silence" else n_classes_timi
    os.environ["WANDB_MODE"] = "offline"
    return_attentions = False
    isGpuEnabled = False
    extract_manual_spec = extract_manual_spec

    return {
        "metadata_file": metadata_file,
        "model_name": model_name,
        "pretrained": isPretrained,
        "lr": lr,
        "decay": decay,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "is_gpu_enabled": isGpuEnabled,
        "mode": mode,
        "is_augment_enabled": is_augment_enabled,
        "is_validation_enabled": isValidationEnabled,
        "dataset": dataset,
        "n_classes": n_classes,
        "epochs": epochs,
        "return_attentions": return_attentions,
        "extract_manual_spec": extract_manual_spec
    }


def get_dataset(dataset_name, dataset_base_path, model_name, partition: Literal["training", "validation", "test"]):
    if dataset_name == "asv19":
        dataset = LoadAsvSpoof19(dataset_base_path, partition, model_name, transform=True,
                                 extract_manual_spec=True)
        dataset_waveform = LoadAsvSpoof19(dataset_base_path, partition, model_name, transform=True,
                                          should_return_waveform=True,
                                          extract_manual_spec=False)
    elif dataset_name == "timi":
        dataset = LoadTimiDataset(base_path=dataset_base_path,
                                  metadata_file_path="clean.csv", partition=partition,
                                  model_name=model_name,
                                  transform=True, is_validation_enabled=True,
                                  extract_manual_spec=True)
        dataset_waveform = LoadTimiDataset(base_path=dataset_base_path,
                                           metadata_file_path="clean.csv", partition=partition,
                                           model_name=model_name,
                                           transform=True, is_validation_enabled=True,
                                           should_return_waveform=True)
    else:
        raise ValueError("Invalid dataset name, only allowed are 'asv19' and 'timi'")
    return dataset, dataset_waveform


def apply_attack(n_samples: int, dataset_len: int, dataset: Dataset, dataset_waveform: Dataset, is_target_attack: bool,
                 is_attack_on_all_classes: bool, should_print_specs: bool):
    n_correct_clean_preds = 0
    n_correct_aug_preds = 0
    for i in range(n_samples):
        if i >= dataset_len:
            break
        audio, spec, audio_waveform, audio_class = extract_dict_vals(i, dataset, dataset_waveform, is_target_attack)
        if not isinstance(audio_waveform, np.ndarray):
            audio_waveform = np.array(audio_waveform)
        sr = 16000
        aug_audio_waveform = asv19_attack_class_based(audio_waveform, audio_class, (2, 1), 40, 0.1, sr=sr)
        # preprocess data by calling manually the dataset method
        aug_audio = preprocess_audio(aug_audio_waveform, sr, original_dataset)

        # print specs to check the effects of the attack
        if i == 1 and should_print_specs:
            print_specs(spec, aug_audio_waveform, sr)

        with torch.no_grad():
            clean_outputs = model(audio.unsqueeze(0))
            aug_outputs = model(aug_audio.unsqueeze(0))

        clean_probabilities = torch.softmax(torch.tensor(clean_outputs), dim=1)
        aug_probabilities = torch.softmax(torch.tensor(aug_outputs), dim=1)
        clean_predicted_class = torch.argmax(clean_probabilities).item()
        aug_predicted_class = torch.argmax(aug_probabilities).item()

        if clean_predicted_class == audio_class:
            n_correct_clean_preds += 1
        if aug_predicted_class == audio_class:
            n_correct_aug_preds += 1
        if not is_attack_on_all_classes:
            print(
                f"Sample #{i + 1} - CLEAN pred:{clean_predicted_class}|p:{torch.max(clean_probabilities).item():.2f}, AUG pred:{aug_predicted_class}|p:{torch.max(aug_probabilities).item():.2f}, Correct:{audio_class}")
    return n_correct_clean_preds, n_correct_aug_preds


def get_target_attack_dataset(n_samples, target_class, original_dataset: Dataset, original_dataset_waveform: Dataset):
    dataset_dt = DataLoader(original_dataset, batch_size=n_samples, shuffle=False, num_workers=0,
                            pin_memory=False)
    dataset_waveform_dt = DataLoader(original_dataset_waveform, batch_size=n_samples, shuffle=False,
                                     num_workers=0,
                                     pin_memory=False)
    dataset = load_specific_classes(dataset_dt, target_class=target_class, n=n_samples)
    dataset_waveform = load_specific_classes(dataset_waveform_dt, target_class=target_class, n=n_samples)
    return dataset, dataset_waveform


if __name__ == "__main__":
    load_dotenv()

    # ------- BEGIN CONFIGURATION -------

    # model_name = "attVgg16"
    model_name = "passt"

    dataset_name = "timi"
    # dataset_name = "asv19"

    partition = "validation"

    N_SAMPLES = 10
    # attack only one target class
    is_target_attack = True
    target_class = 3

    # set True to test the attack on N_SAMPLES of each class, separately
    test_attack_on_all_classes = True
    # plot the spectrogram of the CLEAN and AUGMENTED audio
    should_print_specs = True
    # set true to test the fixed model (i.e. the model trained on the attacked dataset)
    test_fixed_model = False

    # ------- END CONFIGURATION -------

    extract_manual_spec = True
    dataset_base_path = get_dataset_base_path(dataset_name)
    ckpt_path = get_vulnerable_ckpt(model_name=model_name, dataset_name=dataset_name)

    if test_fixed_model:
        ckpt_path = get_fixed_ckpt()

    kwargs = get_model_kwargs(model_name=model_name, extract_manual_spec=extract_manual_spec, dataset=dataset_name,
                              mode="normal", pretrained=False)
    model = SyntheticClassifier.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                     map_location=torch.device(
                                                         "cpu") if not torch.cuda.is_available() else None,
                                                     strict=False,
                                                     **kwargs)
    original_dataset, original_dataset_waveform = get_dataset(dataset_name=dataset_name, model_name=model_name,
                                                              dataset_base_path=dataset_base_path, partition=partition)

    print("*************** BEGIN *********************")
    if test_attack_on_all_classes:
        for cur_class in range(kwargs["n_classes"]):
            target_class = cur_class
            dataset, dataset_waveform = get_target_attack_dataset(n_samples=N_SAMPLES, target_class=target_class,
                                                                  original_dataset=original_dataset,
                                                                  original_dataset_waveform=original_dataset_waveform)
            model.eval()
            dataset_len = dataset[AUDIO_KEY].shape[0] if is_target_attack else len(dataset)
            print(f"*************** BEGIN ON CLASS {cur_class} *********************")
            n_correct_clean_preds, n_correct_aug_preds = apply_attack(N_SAMPLES, dataset_len, dataset, dataset_waveform,
                                                                      is_target_attack,
                                                                      is_attack_on_all_classes=test_attack_on_all_classes,
                                                                      should_print_specs=should_print_specs)
            print(
                f"Correct predictions for class {cur_class}: CLEAN->{n_correct_clean_preds / dataset_len * 100:.3f}%, AUG->{n_correct_aug_preds / dataset_len * 100:.3f}%")
    else:
        # load specific classes
        if is_target_attack:
            dataset, dataset_waveform = get_target_attack_dataset(n_samples=N_SAMPLES, target_class=target_class,
                                                                  original_dataset=original_dataset,
                                                                  original_dataset_waveform=original_dataset_waveform)
        else:
            dataset = original_dataset
            dataset_waveform = original_dataset_waveform

        model.eval()
        dataset_len = dataset[AUDIO_KEY].shape[0] if is_target_attack else len(dataset)
        n_correct_clean_preds, n_correct_aug_preds = apply_attack(N_SAMPLES, dataset_len, dataset, dataset_waveform,
                                                                  is_target_attack,
                                                                  is_attack_on_all_classes=test_attack_on_all_classes,
                                                                  should_print_specs=should_print_specs)
        print(
            f"Correct predictions: CLEAN->{n_correct_clean_preds / N_SAMPLES * 100:.3f}%, AUG->{n_correct_aug_preds / N_SAMPLES * 100:.3f}%")
