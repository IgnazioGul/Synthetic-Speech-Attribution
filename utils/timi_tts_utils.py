import os
from csv import writer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from timi_tts_constants import LABELS_MAP, CSV_METADATA_HEADER, CLASS_KEY


class TimiTtsUtils:

    @staticmethod
    def plot_classes_distribution(root_dir: str, metatada_file: str):
        """
        Plots the distribution of classes in histogram-like style \n
        :param root_dir: root dir of dataset
        :param metatada_file: name of the metadata csv file
        :return:
        """
        metadata = pd.read_csv(os.path.join(root_dir, metatada_file))
        plt.figure(figsize=(15, 6))
        ax = sns.countplot(x=CLASS_KEY, data=metadata)
        ax.bar_label(container=ax.containers[0], labels=LABELS_MAP.keys())
        plt.title("Count of records in each class for " + metatada_file)
        plt.xticks(rotation="vertical")
        plt.show()

    @staticmethod
    def open_label_files(root_dir: str, mode: str) -> dict:
        """
        Returns a dict containing the object of metadata csv files \n
        :param root_dir: root dir of dataset
        :param mode: open mode
        :return: dict containing keys "CLEAN, "AUG", "DTW", "DTW_AUG, "ALL" and respective csv file objects
        """
        clean_labels_file = open(os.path.join(root_dir, "clean.csv"), mode, newline="")
        aug_labels_file = open(os.path.join(root_dir, "aug.csv"), mode, newline="")
        dtw_labels_file = open(os.path.join(root_dir, "dtw.csv"), mode, newline="")
        dtw_aug_labels_file = open(os.path.join(root_dir, "dtw_aug.csv"), mode, newline="")
        all_labels_file = open(os.path.join(root_dir, "all.csv"), mode, newline="")

        files_map = {"CLEAN": clean_labels_file, "AUG": aug_labels_file, "DTW": dtw_labels_file,
                     "DTW_AUG": dtw_aug_labels_file, "ALL": all_labels_file}
        return files_map

    @staticmethod
    def generate_timi_tts_labels(root_dir: str) -> None:
        """
        Generates csv metadata files containing [full_path,label] columns, for each category of data.\n
        The files are: all.csv, aug.csv, clean.csv, dtw.csv, dwt_aug,csv \n
        :param root_dir: root dir of dataset
        :return: None
        """
        partition_2 = ["multi_speaker", "single_speaker"]
        files_map = TimiTtsUtils.open_label_files(root_dir, "w")
        for label_file in files_map.values():
            tmp_w = writer(label_file)
            tmp_w.writerow(CSV_METADATA_HEADER)
        all_files_wr = writer(files_map["ALL"])
        i = 0
        for subdir, dirs, files in os.walk(root_dir):
            # add trailing slash to ease substring search
            subdir = subdir + os.sep
            for file in files:
                # avoid root-level .csv metadata files
                if any(p in subdir for p in partition_2):
                    for data_type in files_map.keys():
                        # get current file data_type (AUG, CLEAN..) from file path
                        if os.sep + data_type + os.sep in subdir:
                            # get current file generator label (gtss (label 1),..) from path
                            for label in LABELS_MAP.keys():
                                if os.sep + label + os.sep in subdir:
                                    wr = writer(files_map[data_type])
                                    # save file path and label in specific metadata file
                                    wr.writerow([os.path.join(subdir, file), LABELS_MAP[label]])
                                    # save file path and label in all metadata file
                                    all_files_wr.writerow([os.path.join(subdir, file), LABELS_MAP[label]])
                                    i += 1
                                    break
        for label_file in files_map.values():
            label_file.close()
        print("Added ", i, " files to label metadata csv")


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    TimiTtsUtils.generate_timi_tts_labels(os.getenv("TIMI-TTS-ROOT-DIR"))
    # TimiTtsUtils.plot_classes_distribution(os.getenv("TIMI-TTS-ROOT-DIR"), "clean.csv")
