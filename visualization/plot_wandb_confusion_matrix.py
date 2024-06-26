import json
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import ConfusionMatrixDisplay


def get_specific_run(api, entity, project_name, run_id):
    run = api.run(f"{entity}/{project_name}/{run_id}")
    return run


if __name__ == '__main__':
    api = wandb.Api()

    # WANDB CONFIGURATION
    entity = ''
    project_name = ''
    run_id = ''

    run = get_specific_run(api, entity, project_name, run_id)

    # Retrieve confusion matrix table artifact info from the run
    conf_matrix_path = run.summary['confusion_matrix_table']['path']

    file = run.file(conf_matrix_path)

    if file is not None:
        with file.download(replace=True) as f:
            file_content = f.read()
        os.remove(f.name)

    conf_matrix = json.loads(file_content)
    columns = conf_matrix['columns']
    data = conf_matrix['data']

    confusion_matrix = {}

    # Convert the data from wandb format for plotting
    for entry in data:
        actual = entry[0]
        predicted = entry[1]
        count = entry[2]
        if actual not in confusion_matrix:
            confusion_matrix[actual] = {}
        confusion_matrix[actual][predicted] = count

    # Determine all unique labels (classes)
    labels = sorted(set(confusion_matrix.keys()).union(
        set(label for label_dict in confusion_matrix.values()
            for label in label_dict.keys())))

    conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Populate the numpy array with counts from the confusion matrix dictionary
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if actual in confusion_matrix and predicted in confusion_matrix[actual]:
                conf_matrix[i, j] = confusion_matrix[actual][predicted]

    display_labels = labels

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=labels
    )
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.show()
