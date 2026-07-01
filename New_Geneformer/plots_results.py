import sys
import os
from geneformer import Classifier

# 1. PATH INJECTION
sys.path.insert(0, "/home/htc/mnakam/miniforge3/envs/geneformer_311/lib/python3.11/site-packages")

# 2. CONFIGURATION
CLF_OUT_DIR = "classifier_output"
CLF_PREFIX = "covid_monocytes"

# 3. INITIALIZE CLASSIFIER (Required to access plotting methods)
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": ["normal", "COVID-19"]},
    filter_data={"cell_type": ["CD14-positive monocyte"]},
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=50,
    nproc=1
)

# 4. PLOT PREDICTIONS
# This creates ROC-AUC and Precision-Recall curves
cc.plot_predictions(
    predictions_file=f"{CLF_OUT_DIR}/predictions.pkl",
    id_class_dict_file=f"{CLF_OUT_DIR}/{CLF_PREFIX}_id_class_dict.pkl",
    title="COVID-19 Monocyte Classification",
    output_directory=CLF_OUT_DIR,
    output_prefix=CLF_PREFIX,
    custom_class_order=["normal", "COVID-19"]
)

print(f"Plots saved to {CLF_OUT_DIR}")