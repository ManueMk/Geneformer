import os
import sys
# Force the script to look at the Conda folder first
sys.path.insert(0, "/home/htc/mnakam/miniforge3/envs/geneformer_311/lib/python3.11/site-packages")
import pickle
import torch
import scanpy as sc
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer, Classifier
import geneformer

# ---------------------------------------------------------
# 1. MONKEY PATCHES (Required for HF Library Compatibility)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. STEP 1: SCANPY PREPROCESSING
# ---------------------------------------------------------
RAW_DATA = "/home/htc/mnakam/mnt/Geneformer/Geneformer/data/Covid19.h5ad"
H5AD_PATH = "data/covid_monocytes_2k.h5ad"

if not os.path.exists(H5AD_PATH):
    print("--- Step 1: Preprocessing with Scanpy ---")
    adata = sc.read_h5ad(RAW_DATA)
    
    # Filter cells
    adata = adata[(adata.obs["cell_type"] == "CD14-positive monocyte") & 
                  (adata.obs["disease"].isin(["COVID-19", "normal"]))]
    
    # Subsample to 1k each
    covid = adata[adata.obs["disease"] == "COVID-19"]
    normal = adata[adata.obs["disease"] == "normal"]
    sc.pp.subsample(covid, n_obs=1000)
    sc.pp.subsample(normal, n_obs=1000)
    
    # Merge
    adata_small = covid.concatenate(normal)
    
    # CRITICAL: Calculate n_counts (Geneformer requirement)
    # This sums the gene counts for each cell
    adata_small.obs["n_counts"] = np.array(adata_small.X.sum(axis=1)).flatten()
    
    # Ensure Ensembl IDs are present
    adata_small.var["ensembl_id"] = adata_small.var_names
    
    # Save
    adata_small.write_h5ad(H5AD_PATH)
    print(f"Saved processed data with n_counts to {H5AD_PATH}")

# ---------------------------------------------------------
# 3. STEP 2: TOKENIZATION
# ---------------------------------------------------------
TOKEN_OUT_DIR = "tokenized_data/"
TOKEN_PREFIX = "tokenized_covid_2k"

if not os.path.exists(f"{TOKEN_OUT_DIR}/{TOKEN_PREFIX}.dataset"):
    print("--- Step 2: Tokenizing ---")
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_type": "cell_type", "disease": "disease", "donor_id": "donor_id"},
        nproc=4
    )
    tk.tokenize_data("data/", TOKEN_OUT_DIR, TOKEN_PREFIX, file_format="h5ad")

# ---------------------------------------------------------
# 4. STEP 3: PREPARE DATA FOR CLASSIFIER
# ---------------------------------------------------------
CLF_OUT_DIR = "classifier_output"
CLF_PREFIX = "covid_monocytes"

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease", "states": ["normal", "COVID-19"]},
    filter_data={"cell_type": ["CD14-positive monocyte"]},
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=50,
    nproc=1
)

if not os.path.exists(f"{CLF_OUT_DIR}/{CLF_PREFIX}_labeled_train.dataset"):
    print("--- Step 3: Labeling Data ---")
    cc.prepare_data(
        input_data_file=f"{TOKEN_OUT_DIR}/{TOKEN_PREFIX}.dataset",
        output_directory=CLF_OUT_DIR,
        output_prefix=CLF_PREFIX
    )

# ---------------------------------------------------------
# 5. STEP 4: TRAINING
# ---------------------------------------------------------
print("--- Step 4: Training ---")
train_data = load_from_disk(f"{CLF_OUT_DIR}/{CLF_PREFIX}_labeled_train.dataset")
eval_data = load_from_disk(f"{CLF_OUT_DIR}/{CLF_PREFIX}_labeled_test.dataset")

with open(f"{CLF_OUT_DIR}/{CLF_PREFIX}_id_class_dict.pkl", "rb") as f:
    id_class_dict = pickle.load(f)

# Using V2-104M as per your request
model = cc.train_classifier(
    model_directory="/home/htc/mnakam/mnt/Geneformer/Geneformer/pretrained_models/Geneformer/Geneformer-V2-104M",
    num_classes=len(id_class_dict),
    train_data=train_data,
    eval_data=eval_data,
    output_directory=CLF_OUT_DIR,
    predict=True
)

print("--- Pipeline Complete! ---")