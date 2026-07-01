import sys
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from geneformer import Classifier
import geneformer

# 1. PATH INJECTION
sys.path.insert(0, "/home/htc/mnakam/miniforge3/envs/geneformer_311/lib/python3.11/site-packages")

# 2. MONKEY PATCHES (Keep these to prevent the TypeError/AttributeError)
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.padding_side = "right"
    def pad(self, encoded_inputs, **kwargs):
        return encoded_inputs

def fixed_cell_collator_init(self, token_dictionary):
    self.token_dictionary = token_dictionary
    self.padding = True
    self.max_length = 2048
    self.classifier = "cell"
    self.tokenizer = MockTokenizer()

def fixed_cell_collator_call(self, features):
    if isinstance(features[0], dict):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [f["label"] for f in features]
    else:
        input_ids = [torch.tensor(f[0]) for f in features]
        labels = [f[-1] for f in features]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded_input_ids != 0).long()
    return {"input_ids": padded_input_ids, "attention_mask": attention_mask, "labels": torch.tensor(labels, dtype=torch.long)}

geneformer.collator_for_classification.DataCollatorForCellClassification.__init__ = fixed_cell_collator_init
geneformer.collator_for_classification.DataCollatorForCellClassification.__call__ = fixed_cell_collator_call

# 3. CONFIGURATION
# Based on your 'ls' output, the model files are directly in this folder:
CLF_OUT_DIR = "classifier_output"
CLF_PREFIX = "covid_monocytes"

# 4. INITIALIZE CLASSIFIER
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": ["normal", "COVID-19"]},
    filter_data={"cell_type": ["CD14-positive monocyte"]},
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=50,
    nproc=1
)

# 5. STEP 5: EVALUATION
print("--- Starting Evaluation ---")

all_metrics = cc.validate(
    model_directory=CLF_OUT_DIR,  # Points to folder containing model.safetensors
    prepared_input_data_file=f"{CLF_OUT_DIR}/{CLF_PREFIX}_labeled_test.dataset",
    id_class_dict_file=f"{CLF_OUT_DIR}/{CLF_PREFIX}_id_class_dict.pkl",
    output_directory=CLF_OUT_DIR,
    output_prefix=CLF_PREFIX,
    predict_eval=True
)

# 6. STEP 6: PLOT CONFUSION MATRIX
print("--- Plotting Confusion Matrix ---")

cc.plot_conf_mat(
    conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
    output_directory=CLF_OUT_DIR,
    output_prefix=CLF_PREFIX,
    custom_class_order=["normal", "COVID-19"]
)

# 7. STEP 7: PLOT  ROC/PR CURVES

print("--- Generating ROC and Precision-Recall Curves ---")

# This function creates the ROC curve and the PR curve
cc.plot_predictions(
    predictions_file=f"{CLF_OUT_DIR}/predictions.pkl",
    id_class_dict_file=f"{CLF_OUT_DIR}/{CLF_PREFIX}_id_class_dict.pkl",
    title="COVID-19 Classification Performance",
    output_directory=CLF_OUT_DIR,
    output_prefix=CLF_PREFIX,
    custom_class_order=["normal", "COVID-19"]
)

print(f"Success! Evaluation complete. Plot saved in {CLF_OUT_DIR}")