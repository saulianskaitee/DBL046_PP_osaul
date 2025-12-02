#!/usr/bin/env python

import os
import sys
import gzip
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import esm
from esm.inverse_folding.util import (
    load_structure,
    extract_coords_from_structure,
    get_encoder_output,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

# =========================
# Config
# =========================

dataframe_path = "/work3/s232958/data/PPint_DB/disordered_interfaces_no_cutoff_filtered_nonredundant80_3å_5.csv.gz"
download_dir = "/work3/s232958/data/PPint_DB/pdb_cache"
name_column = "PDB_chain_name"
sequence_column = "sequence"

# Where to save ESM-IF embeddings (.npy per chain)
path_to_output_embeddings = "/work3/s232958/data/PPint_DB/esmif_embeddings_full"

# Root of your local PDB mirror (with subfolders a0, a1, etc.)
root_pdb_dir = "/novo/users/cpjb/rdd/PDB_mirror_pdb"

import requests

def download_pdb(pdb_id: str, out_dir: str) -> str:
    """
    Download PDB file (.pdb) from RCSB if not already cached.
    Returns path to the downloaded file.
    """
    pdb_id = pdb_id.lower()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb.gz")

    if os.path.exists(out_path):
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from RCSB...", file=sys.stderr)

    r = requests.get(url)
    if r.status_code != 200:
        raise FileNotFoundError(f"Failed to download {pdb_id} from RCSB")

    # Save gzipped version for compatibility with your loader
    with gzip.open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def index_pdb_files(root_dir: str):
    """
    Walk the PDB mirror and build:
        pdb_id (4-letter, lowercase) -> full path to .pdb.gz

    Adjust filename parsing if your mirror uses a different naming scheme.
    """
    pdb_index = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            fn_lower = fn.lower()
            if not fn_lower.endswith(".pdb.gz"):
                continue

            # strip ".pdb.gz"
            stem = fn_lower[:-7]

            # Common naming patterns:
            # - "1abc.pdb.gz"           -> pdb_id = "1abc"
            # - "pdb1abc.ent.gz" etc.   -> adjust below if needed
            if stem.startswith("pdb") and len(stem) >= 7:
                # e.g., "pdb1abc" -> "1abc"
                pdb_id = stem[3:7]
            else:
                # assume first 4 chars are the PDB id
                pdb_id = stem[:4]

            pdb_index[pdb_id] = os.path.join(dirpath, fn)

    return pdb_index


def load_structure_from_gz(path_gz: str, chain_id: str):
    """
    Decompress a .pdb.gz to a temporary file, load with esm's load_structure,
    and return the Structure object.
    """
    if not os.path.isfile(path_gz):
        raise FileNotFoundError(f"PDB file not found: {path_gz}")

    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        with gzip.open(path_gz, "rb") as fin, open(tmp.name, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        structure = load_structure(tmp.name, chain=chain_id)

    return structure


def calculate_esm_if_embeddings(model, alphabet, pdb_path: str, chain_id: str) -> np.ndarray:
    structure = load_structure_from_gz(pdb_path, chain_id)
    coords, _seq = extract_coords_from_structure(structure)

    # NEW: move coords to same device as model
    coords = coords.to(next(model.parameters()).device)

    with torch.no_grad():
        rep = get_encoder_output(model, alphabet, coords)

    return rep.cpu().numpy()


# =========================
# Main
# =========================

def main():

    # Check if directories exist and create them if not
    os.makedirs(path_to_output_embeddings, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # ---- Load dataframe ----
    print(f"Reading dataframe from: {dataframe_path}", file=sys.stderr)
    sequence_df = pd.read_csv(dataframe_path)

    # Build unique chain name: "PDB_chainname" -> e.g. "1ABC_A"
    sequence_df[name_column] = (sequence_df["PDB"] + "_" + sequence_df["chainname"]).tolist()

    # Basic cleaning
    sequence_df = sequence_df[sequence_df[sequence_column].notna()]
    sequence_df = sequence_df.drop_duplicates(subset=[name_column, sequence_column])

    # Directory to store downloaded structures
    sequence_df["pdb_path"] = sequence_df["PDB"].str.lower().apply(lambda pid: download_pdb(pid, download_dir))

    # Drop rows where we don't have a PDB file
    missing_mask = sequence_df["pdb_path"].isna()
    if missing_mask.any():
        missing_ids = sequence_df.loc[missing_mask, "PDB"].unique()
        print("Warning: no PDB file found for these IDs:", file=sys.stderr)
        for pid in missing_ids:
            print(f"  {pid}", file=sys.stderr)
        sequence_df = sequence_df[~missing_mask]

    print(f"Remaining rows after filtering: {len(sequence_df)}", file=sys.stderr)

    # ---- Load ESM-IF model ----
    print("Loading ESM-IF1 model...", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()
    print(f"Model loaded on device: {device}", file=sys.stderr)

    # ---- Set of already computed embeddings ----
    already_calculated_files = {
        fname[:-4] for fname in os.listdir(path_to_output_embeddings) if fname.endswith(".npy")
    }
    print(f"Found {len(already_calculated_files)} existing .npy files", file=sys.stderr)

    # ---- Main loop ----
    for idx, row in tqdm(sequence_df.iterrows(), total=len(sequence_df)):
        name = row[name_column]     # e.g. "1ABC_A"
        if name in already_calculated_files:
            continue

        pdb_path = row["pdb_path"]
        chain_id = row["chainname"]  # e.g. "A"
        try:
            embeddings = calculate_esm_if_embeddings(model, alphabet, pdb_path, chain_id)
        except Exception as e:
            print(f"Failed for {name}: {e}", file=sys.stderr)
            continue

        out_path = os.path.join(path_to_output_embeddings, name + ".npy")
        np.save(out_path, embeddings)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main() 