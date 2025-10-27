#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd
import random
import torch
from tqdm import trange

PPint_interactions = pd.read_csv(
    "data/PPint_DB/disordered_interfaces_no_cutoff_filtered_nonredundant80_3å_5.csv.gz",
    index_col=0
).reset_index(drop=True)

# build dict of interface -> {seq_target, seq_binder}
PPint_interactions_dict = {}
for idx, row in PPint_interactions.iterrows():
    pdb_inter_name = row["PDB_interface_name"]
    seq = row["sequence"]
    if pdb_inter_name not in PPint_interactions_dict:
        PPint_interactions_dict[pdb_inter_name] = {
            "seq_target": seq,
            "seq_binder": "placeholder"
        }
    else:
        PPint_interactions_dict[pdb_inter_name]["seq_binder"] = seq

# dict -> df
PPint_interactions_df_NEW = pd.DataFrame([
    {
        "target_id": inter_name,
        "seq_target": vals["seq_target"],
        "seq_binder": vals["seq_binder"],
    }
    for inter_name, vals in PPint_interactions_dict.items()
])

# map unique sequences to stable ids
target_seq_to_id = {}
binder_seq_to_id = {}

for _, row in PPint_interactions_df_NEW.iterrows():
    seq_t = row["seq_target"]
    seq_b = row["seq_binder"]
    tid = row["target_id"]

    if seq_t not in target_seq_to_id:
        target_seq_to_id[seq_t] = tid
    if seq_b not in binder_seq_to_id:
        binder_seq_to_id[seq_b] = tid

PPint_interactions_df_NEW["target_seq_to_id"] = PPint_interactions_df_NEW["seq_target"].map(target_seq_to_id)
PPint_interactions_df_NEW["binder_seq_to_id"] = PPint_interactions_df_NEW["seq_binder"].map(binder_seq_to_id)

PPint_interactions_df_NEW["target_binder_id"] = (
    PPint_interactions_df_NEW["target_seq_to_id"] + "_" +
    PPint_interactions_df_NEW["binder_seq_to_id"]
)

PPint_interactions_df_NEW = PPint_interactions_df_NEW.drop(columns=["target_id"]).rename(
    columns={
        "target_seq_to_id": "target_id",
        "binder_seq_to_id": "binder_id"
    }
)

# sample subset
PPint_interactions_df_NEW_sample = PPint_interactions_df_NEW.sample(
    n=3532,
    random_state=0
).reset_index(drop=True)

# load esm2 model
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model.eval().to("cuda")
batch_converter = alphabet.get_batch_converter()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def _generate_masked_sequences(sequence: str, mask_length: int):
    np.random.seed(0)
    seq_indexes = np.arange(0, len(sequence))
    np.random.shuffle(seq_indexes)
    batched_seq_indexes = list(batch(seq_indexes, n=mask_length))
    all_sequences = []
    for masked_index in batched_seq_indexes:
        seq_copy = list(sequence).copy()
        for index in masked_index:
            seq_copy[index] = "<mask>"
        all_sequences.append((masked_index, "".join(seq_copy)))
    return all_sequences

@torch.no_grad()
def calculate_pll_score(sequence: str, mask_length: int = 1):
    if not sequence or not isinstance(sequence, str):
        raise ValueError("Input sequence must be a non-empty string.")

    np.random.seed(0)
    masked_data = _generate_masked_sequences(sequence, mask_length)

    # Build ESM input for ALL masked variants in this sequence
    ESM_input = [(i, masked_seq[1]) for i, masked_seq in enumerate(masked_data)]

    batch_labels, batch_strs, batch_tokens = batch_converter(ESM_input)
    batch_tokens = batch_tokens.to("cuda")

    out = model(batch_tokens, repr_layers=[33], return_contacts=False)
    logits = out["logits"]
    logit_prob = torch.nn.functional.log_softmax(logits, dim=-1)

    log_likelihood = 0.0
    for i, (masked_index, _) in enumerate(masked_data):
        for j in masked_index:
            log_likelihood += logit_prob[i, j+1, alphabet.get_idx(sequence[j])]

    # cleanup GPU to control memory growth
    del out, logits, logit_prob, batch_tokens
    torch.cuda.empty_cache()

    avg_log_likelihood = log_likelihood / len(sequence)
    pll = float(torch.exp(-torch.tensor(avg_log_likelihood)).item())

    return float(avg_log_likelihood), pll

# iterate through sampled binders
for i in trange(len(PPint_interactions_df_NEW_sample)):
    seq = PPint_interactions_df_NEW_sample.iloc[i]["seq_binder"]
    avg_log_likelihood, pll = calculate_pll_score(sequence=seq, mask_length=1)
    PPint_interactions_df_NEW_sample.at[i, "binder_avgLL"] = avg_log_likelihood
    PPint_interactions_df_NEW_sample.at[i, "pseudo_perplexity"] = pll

    if i % 100 == 0:
        print(f"[{i}] avgLL={avg_log_likelihood:.4f}, pll={pll:.4f}")
        # print_mem_consumption()  # <- comment out or define this yourself

# Save results (optional)
PPint_interactions_df_NEW_sample.to_csv("data/PPint_DB/averageLL_pPLL_PPint.csv", index=False)
print("Done. Saved to averageLL_pPLL.csv")
