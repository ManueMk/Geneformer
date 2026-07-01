import sys
import os
import pickle
import torch
import geneformer.in_silico_perturber_stats as isps
from geneformer import InSilicoPerturberStats

# 1. PATH INJECTION
sys.path.insert(0, "/home/htc/mnakam/miniforge3/envs/geneformer_311/lib/python3.11/site-packages")

# ---------------------------------------------------------
# 2. MONKEY PATCH: Fixes "TypeError: argument of type 'int' is not iterable"
# ---------------------------------------------------------
def patched_read_dict(cos_sims_dict, cell_or_gene_emb, anchor_token):
    """Revised internal Geneformer function to handle numeric keys safely."""
    # We use str(k) so that it doesn't crash if k is an integer
    cell_emb_dict = {
        k: v for k, v in cos_sims_dict.items() if v and "cell_emb" in str(k)
    }
    gene_emb_dict = {
        k: v for k, v in cos_sims_dict.items() if v and "gene_emb" in str(k)
    }
    
    if cell_or_gene_emb == "cell":
        return [cell_emb_dict]
    elif cell_or_gene_emb == "gene":
        return [gene_emb_dict]
    elif cell_or_gene_emb == "both":
        return [cell_emb_dict, gene_emb_dict]

# Inject the fix into the loaded geneformer module
isps.read_dict = patched_read_dict
# ---------------------------------------------------------

# 3. CONFIGURATION
OUTPUT_DIR = "perturb_output"
PREFIX = "covid_monocytes"
CLF_OUT_DIR = "classifier_output"

# Load the IDs (Ensures we use 1 for COVID and 0 for normal)
with open(f"{CLF_OUT_DIR}/covid_monocytes_id_class_dict.pkl", "rb") as f:
    id_class_dict = pickle.load(f)
name_to_id = {v: k for k, v in id_class_dict.items()}

covid_id = name_to_id["COVID-19"] # Usually 1
normal_id = name_to_id["normal"]   # Usually 0

# 4. INITIALIZE STATS
print(f"--- Initializing Stats with start_state={covid_id} and goal_state={normal_id} ---")
ispstats = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed="all",
    cell_states_to_model={
        "state_key": "label",      # Use 'label' because that is the column name
        "start_state": covid_id,   # Use the integer ID (1)
        "goal_state": normal_id    # Use the integer ID (0)
    }
)

# 5. RUN STATS
print("--- Calculating Perturbation Stats (using numeric IDs) ---")
ispstats.get_stats(
    input_data_directory=OUTPUT_DIR,
    null_dist_data_directory=None,
    output_directory=OUTPUT_DIR,
    output_prefix=PREFIX
)

print(f"Success! Check the CSV files in {OUTPUT_DIR}")