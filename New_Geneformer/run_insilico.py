import sys
import os
import torch
import pickle
import geneformer
from geneformer import InSilicoPerturber, InSilicoPerturberStats

# 1. PATH INJECTION
sys.path.insert(0, "/home/htc/mnakam/miniforge3/envs/geneformer_311/lib/python3.11/site-packages")

# 2. MONKEY PATCH (Required)
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

geneformer.collator_for_classification.DataCollatorForCellClassification.__init__ = fixed_cell_collator_init

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 3. IDENTIFY IDs FROM DICTIONARY
    # ---------------------------------------------------------
    CLF_OUT_DIR = "classifier_output"
    with open(f"{CLF_OUT_DIR}/covid_monocytes_id_class_dict.pkl", "rb") as f:
        id_class_dict = pickle.load(f)

    # We find the IDs by looking up the names in the pkl
    # id_class_dict looks like: {0: 'normal', 1: 'COVID-19'}
    # We swap them to find the ID by name:
    name_to_id = {v: k for k, v in id_class_dict.items()}
    
    covid_id = name_to_id["COVID-19"] # This will be 1
    normal_id = name_to_id["normal"]   # This will be 0

    # ---------------------------------------------------------
    # 4. CONSTRUCT THE MATCHING DICTIONARIES
    # ---------------------------------------------------------
    # The keys in state_embs_dict MUST match the start/goal states
    state_embs_dict = {
        covid_id: torch.tensor([covid_id]),
        normal_id: torch.tensor([normal_id])
    }

    cell_states_to_model = {
        "state_key": "label",      # The column name in the dataset
        "start_state": covid_id,   # The numeric ID for COVID (1)
        "goal_state": normal_id    # The numeric ID for Normal (0)
    }

    print(f"Environment Check: covid_id={covid_id}, normal_id={normal_id}")
    print(f"state_embs_dict: {state_embs_dict}")
    print(f"cell_states_to_model: {cell_states_to_model}")

    # ---------------------------------------------------------
    # 5. INITIALIZE PERTURBER
    # ---------------------------------------------------------
    isp = InSilicoPerturber(
        perturb_type="delete",
        model_type="CellClassifier",
        num_classes=2,
        emb_mode="cls",
        max_ncells=200, 
        cell_states_to_model=cell_states_to_model,
        state_embs_dict=state_embs_dict,
        filter_data={"cell_type": ["CD14-positive monocyte"]},
        forward_batch_size=20,
        nproc=4
    )

    # ---------------------------------------------------------
    # 6. RUN PERTURBATION
    # ---------------------------------------------------------
    OUTPUT_DIR = "perturb_output"
    TRAINED_MODEL = "classifier_output" 
    INPUT_DATA = f"{CLF_OUT_DIR}/covid_monocytes_labeled_test.dataset"

    print("--- Starting In Silico Perturbation ---")
    isp.perturb_data(
        model_directory=TRAINED_MODEL,
        input_data_file=INPUT_DATA,
        output_directory=OUTPUT_DIR,
        output_prefix="covid_monocytes"
    )

    # ---------------------------------------------------------
    # 7. RUN STATS
    # ---------------------------------------------------------
    print("--- Calculating Perturbation Stats ---")
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed="all"
    )

    ispstats.get_stats(
        input_data_directory=OUTPUT_DIR,
        output_directory=OUTPUT_DIR,
        output_prefix="covid_monocytes"
    )