#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def compute_esm2_embeddings(model, batch_tokens, seq):
    """Compute per-residue embeddings using ESM2."""
    with torch.no_grad():
        L = model.num_layers
        reps = model(batch_tokens.to("cuda"), repr_layers=[L], return_contacts=False )["representations"][L][0]
        # assert (len(seq) == reps.shape[0])
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

        train_path = "/work3/s232958/data/PPint_DB/PPint_train_w_pbd_lens.csv"
        test_path  = "/work3/s232958/data/PPint_DB/PPint_test_w_pbd_lens.csv"

        print("Loading datasets...")
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        interaction_dict = {}

        for df in [train_df, test_df]:
            for _, row in df.iterrows():
                key1 = row["ID1"].split("_")[0] + "_" + row["ID1"].split("_")[-1]
                key2 = row["ID2"].split("_")[0] + "_" + row["ID2"].split("_")[-1]

                interaction_dict[key1] = row["seq_pdb_target"]
                interaction_dict[key2] = row["seq_pdb_binder"]

        out_dir = "/work3/s232958/data/PPint_DB/embeddings_esm2"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving embeddings to: {out_dir}")

    elif dataset == "meta":

        meta_path = "/work3/s232958/data/meta_analysis/interaction_df_metaanal.csv"
        print("Loading datasets...")
        meta_df = pd.read_csv(meta_path)

        interaction_dict_B = {}
        interaction_dict_T = {}

        for _, row in meta_df.iterrows():
            binder_key = row.target_binder_ID
            binder_seq = row.A_seq
            target_key = row.target_id_mod
            target_seq = row.B_seq

            interaction_dict_B[binder_key] = binder_seq
            if target_key not in interaction_dict_T:
                interaction_dict_T[target_key] = target_seq

        out_dir_B = "/work3/s232958/data/meta_analysis/embeddings_esm2_binders"
        out_dir_T = "/work3/s232958/data/meta_analysis/embeddings_esm2_targets"

        os.makedirs(out_dir_B, exist_ok=True)
        os.makedirs(out_dir_T, exist_ok=True)

        print(f"Saving binder embeddings to: {out_dir_B}")
        print(f"Saving target embeddings to: {out_dir_T}")

    elif dataset == "boltzgen":

        boltz_path = "/work3/s232958/data/boltzgen/boltzgen_df_filtered.csv"
        print("Loading datasets...")
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

        out_dir = "/work3/s232958/data/boltzgen/embeddings_esm2"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving embeddings to: {out_dir}")

    elif dataset == "bindcraft":

        bindcraft_path = "/work3/s232958/data/bindcraft/bindcraft_with_target_seq.csv"
        print("Loading boltzgen dataset...")
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

        out_dir = "/work3/s232958/data/bindcraft/embeddings_esm2"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving embeddings to: {out_dir}")

    # -----------------------------------------------------
    # Load ESM2 model
    # -----------------------------------------------------
    print("Loading ESM2 model...")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model.eval().to("cuda")
    batch_converter = alphabet.get_batch_converter()
    print("ESM2 model loaded successfully.")

    # -----------------------------------------------------
    # Embedding loops
    # -----------------------------------------------------
    if dataset == "meta":

        print(f"Embedding {len(interaction_dict_B)} binders...")
        for name, seq in tqdm(interaction_dict_B.items()):
            data = [(name, seq)]
            _, _, batch_tokens = batch_converter(data)
            try:
                emb = compute_esm2_embeddings(model, batch_tokens, seq)
                np.save(os.path.join(out_dir_B, f"{name}.npy"), emb)
            except Exception as e:
                print(f"[ERROR] Failed binder {name}: {e}", file=sys.stderr)

        print(f"Embedding {len(interaction_dict_T)} targets...")
        for name, seq in tqdm(interaction_dict_T.items()):
            data = [(name, seq)]
            _, _, batch_tokens = batch_converter(data)
            try:
                emb = compute_esm2_embeddings(model, batch_tokens, seq)
                np.save(os.path.join(out_dir_T, f"{name}.npy"), emb)
            except Exception as e:
                print(f"[ERROR] Failed target {name}: {e}", file=sys.stderr)

    else:
        print(f"Embedding {len(interaction_dict)} sequences...")
        for name, seq in tqdm(interaction_dict.items()):
            data = [(name, seq)]
            _, _, batch_tokens = batch_converter(data)
            try:
                emb = compute_esm2_embeddings(model, batch_tokens, seq)
                np.save(os.path.join(out_dir, f"{name}.npy"), emb)
            except Exception as e:
                print(f"[ERROR] Failed embedding {name}: {e}", file=sys.stderr)

    print("All embeddings computed and saved.")

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()