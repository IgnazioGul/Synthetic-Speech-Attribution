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
    def open_pandas_label_csv(root_dir: str):
        """
        Returns a dict containing the object of metadata csv files with pandas \n
        :param root_dir: root dir of dataset
        :param mode: open mode
        :return: dict containing keys "CLEAN, "AUG", "DTW", "DTW_AUG, "ALL" and respective csv file objects
        """
        clean_labels_csv = pd.read_csv(os.path.join(root_dir, "clean.csv"))
        aug_labels_csv = pd.read_csv(os.path.join(root_dir, "aug.csv"))
        dtw_labels_csv = pd.read_csv(os.path.join(root_dir, "dtw.csv"))
        dtw_aug_labels_csv = pd.read_csv(os.path.join(root_dir, "dtw_aug.csv"))
        all_labels_csv = pd.read_csv(os.path.join(root_dir, "all.csv"))

        return {"CLEAN": clean_labels_csv, "AUG": aug_labels_csv, "DTW": dtw_labels_csv,
                "DTW_AUG": dtw_aug_labels_csv, "ALL": all_labels_csv}

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

    @staticmethod
    def generate_timi_tts_reduced_labels(root_dir: str):
        """
        Generates reduced (2 classes only) csv metadata files containing [full_path,label] columns, for each category of data.\n
        The files are: all.csv, aug.csv, clean.csv, dtw.csv, dwt_aug,csv \n
        :param root_dir: root dir of dataset
        :return: None
        """
        i = 0
        csv_files = TimiTtsUtils.open_pandas_label_csv(root_dir)
        for csv_type, csv_file in csv_files.items():
            new_filename = csv_type.lower() + "_reduced.csv"
            reduced_csv = open(os.path.join(root_dir, new_filename), mode="w", newline="")
            reduced_csv_w = writer(reduced_csv)
            reduced_csv_w.writerow(CSV_METADATA_HEADER)

            reduced_data_2 = csv_file.loc[csv_file[CLASS_KEY] == 2]
            reduced_data_3 = csv_file.loc[csv_file[CLASS_KEY] == 3]
            reduced_data_2[CLASS_KEY] = 0
            reduced_data_3[CLASS_KEY] = 1

            reduced_data = pd.concat([reduced_data_3, reduced_data_2], ignore_index=True)
            reduced_data.to_csv(os.path.join(root_dir, new_filename), index=False)
            # reduced_csv_w.writerows(reduced_data_2)
            # reduced_csv_w.writerows(reduced_data_3)
            i += len(reduced_data_2) + len(reduced_data_3)
            reduced_csv.close()
            print(
                "Added in " + new_filename + " " + str(
                    len(reduced_data_2)) + " files for class 2 (new class 0), and " + str(
                    len(reduced_data_3)) + " files for class 3 (new class 1)")
        print("Added " + str(i) + " files to reduced label metadata csv")


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    TimiTtsUtils.generate_timi_tts_labels(os.getenv("TIMI-TTS-ROOT-DIR"))
    # TimiTtsUtils.plot_classes_distribution(os.getenv("TIMI-TTS-ROOT-DIR"), "clean.csv")
