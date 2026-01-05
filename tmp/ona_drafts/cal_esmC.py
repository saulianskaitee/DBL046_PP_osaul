#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ESMC
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

dataset = "PPint"
# dataset = "meta"
# dataset = "boltzgen"
# dataset = "bindcraft"

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def to_numpy(x):
    """Safely convert torch tensor to numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def compute_esmc_embeddings(client, sequence):
    """Compute per-residue embeddings using ESMC."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    logits_out = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
    reps = logits_out.embeddings[1:-1,:]
    assert (len(sequence) == reps.shape[0])
    return to_numpy(reps)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("Python executable:", sys.executable)
    print("Current working directory:", os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------
    if dataset == "PPint":

        train_path = "/work3/s232958/data/PPint_DB/PPint_train.csv"
        test_path  = "/work3/s232958/data/PPint_DB/PPint_test.csv"

        print("Loading PPint datasets...")
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        interaction_dict = {}

        for df in [train_df, test_df]:
            for _, row in df.iterrows():
                key1 = row["ID1"].split("_")[0] + "_" + row["ID1"].split("_")[-1]
                key2 = row["ID2"].split("_")[0] + "_" + row["ID2"].split("_")[-1]
                interaction_dict[key1] = row["seq_target"]
                interaction_dict[key2] = row["seq_binder"]

        out_dir = "/work3/s232958/data/PPint_DB/embeddings_esmC"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving embeddings to: {out_dir}")

    # -----------------------------------------------------
    # META DATASET (separate binders / targets)
    # -----------------------------------------------------
    elif dataset == "meta":

        meta_path = "/work3/s232958/data/meta_analysis/interaction_df_metaanal.csv"
        print("Loading meta dataset...")
        meta_df = pd.read_csv(meta_path)

        binder_dict = {}
        target_dict = {}

        for _, row in meta_df.iterrows():
            binder_key = row.target_binder_ID
            binder_seq = row.A_seq
            target_key = row.target_id_mod
            target_seq = row.B_seq

            binder_dict[binder_key] = binder_seq

            if target_key not in target_dict:
                target_dict[target_key] = target_seq

        # Output dirs
        out_dir_binders = "/work3/s232958/data/meta_analysis/embeddings_esmC_binders"
        out_dir_targets = "/work3/s232958/data/meta_analysis/embeddings_esmC_targets"

        os.makedirs(out_dir_binders, exist_ok=True)
        os.makedirs(out_dir_targets, exist_ok=True)

        print(f"Binder embeddings → {out_dir_binders}")
        print(f"Target embeddings → {out_dir_targets}")

    # -----------------------------------------------------
    # BOLTZGEN
    # -----------------------------------------------------
    elif dataset == "boltzgen":

        boltz_path = "/work3/s232958/data/boltzgen/boltzgen_df_filtered.csv"
        print("Loading boltzgen dataset...")
        boltz_df = pd.read_csv(boltz_path)

        interaction_dict = {}
        for _, row in boltz_df.iterrows():
            tkey = row.target_id
            tseq = row.target_seq
            bkey = row.binder_id2
            bseq = row.binder_seq

            interaction_dict[bkey] = bseq
            if tkey not in interaction_dict:
                interaction_dict[tkey] = tseq

        out_dir = "/work3/s232958/data/boltzgen/embeddings_esmC"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving embeddings to: {out_dir}")


    elif dataset == "bindcraft":

        bindcraft_path = "/work3/s232958/data/bindcraft/bindcraft_with_target_seq.csv"
        print("Loading bindcraft dataset...")
        bindcraft_df = pd.read_csv(bindcraft_path)

        interaction_dict = {}
        for _, row in bindcraft_df.iterrows():
            tkey = row.target_id
            tseq = row.seq_target
            bkey = row.binder_id
            bseq = row.seq_binder

            interaction_dict[bkey] = bseq
            if tkey not in interaction_dict:
                interaction_dict[tkey] = tseq

        out_dir = "/work3/s232958/data/bindcraft/embeddings_esmC"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving embeddings to: {out_dir}")

    # -----------------------------------------------------
    # Load ESMC model
    # -----------------------------------------------------
    print("Loading ESMC model...")
    client = ESMC.from_pretrained("esmc_600m", device=device)
    print("ESMC model loaded successfully.")

    # -----------------------------------------------------
    # Embedding loops
    # -----------------------------------------------------

    # -------------- META: separate dictionaries --------------
    if dataset == "meta":

        print(f"Embedding {len(binder_dict)} binders...")
        for name, seq in tqdm(binder_dict.items()):
            try:
                emb = compute_esmc_embeddings(client, seq)
                np.save(os.path.join(out_dir_binders, f"{name}.npy"), emb)
            except Exception as e:
                print(f"[ERROR] Failed binder {name}: {e}", file=sys.stderr)

        print(f"Embedding {len(target_dict)} targets...")
        for name, seq in tqdm(target_dict.items()):
            try:
                emb = compute_esmc_embeddings(client, seq)
                np.save(os.path.join(out_dir_targets, f"{name}.npy"), emb)
            except Exception as e:
                print(f"[ERROR] Failed target {name}: {e}", file=sys.stderr)

    # -------------- PPint and Boltzgen (single dict) --------------
    else:
        print(f"Embedding {len(interaction_dict)} sequences...")
        for name, seq in tqdm(interaction_dict.items()):
            try:
                emb = compute_esmc_embeddings(client, seq)
                np.save(os.path.join(out_dir, f"{name}.npy"), emb)
            except Exception as e:
                print(f"[ERROR] Failed embedding {name}: {e}", file=sys.stderr)

    print("All embeddings computed and saved.")

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
