import sys
from geneformer import Classifier

# 1. Path to your conda environment libraries
sys.path.insert(0, "/home/htc/mnakam/miniforge3/envs/geneformer_311/lib/python3.11/site-packages")

# 2. Setup paths
CLF_OUT_DIR = "classifier_output"
CLF_PREFIX = "covid_monocytes"

# 3. Initialize Classifier (to access the plotting tool)
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": ["normal", "COVID-19"]},
    filter_data={"cell_type": ["CD14-positive monocyte"]},
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=50,
    nproc=1
)

# 4. Generate ROC and PR curves
print("Generating plots...")
cc.plot_predictions(
    predictions_file=f"{CLF_OUT_DIR}/predictions.pkl",
    id_class_dict_file=f"{CLF_OUT_DIR}/{CLF_PREFIX}_id_class_dict.pkl",
    title="COVID-19 Classification Performance",
    output_directory=CLF_OUT_DIR,
    output_prefix=CLF_PREFIX,
    custom_class_order=["normal", "COVID-19"]
)
print(f"Plots saved in {CLF_OUT_DIR}")