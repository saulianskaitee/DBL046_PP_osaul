#!/usr/bin/env python

import os
import sys
import gzip
import shutil
import tempfile
import warnings
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import esm
from esm.inverse_folding.util import (
    load_structure,
    extract_coords_from_structure,
    CoordBatchConverter,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------
# Download PDB from RCSB
# ---------------------------------------------------------
def download_pdb(pdb_id: str, out_dir: str) -> str:
    """
    Download PDB file (.pdb) from RCSB if not already cached.
    Returns path to a gzipped file.
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

    with gzip.open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


# ---------------------------------------------------------
# Load structure utility
# ---------------------------------------------------------
def load_structure_from_gz(gz_path: str, chain_id: str):
    if not os.path.isfile(gz_path):
        raise FileNotFoundError(f"PDB file not found: {gz_path}")

    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        with gzip.open(gz_path, "rb") as fin, open(tmp.name, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        structure = load_structure(tmp.name, chain=chain_id)

    return structure


# ---------------------------------------------------------
# Compute embeddings
# ---------------------------------------------------------
def calculate_esm_if_embeddings(model, alphabet, pdb_path: str, chain_id: str) -> np.ndarray:
    """Return per-residue ESM-IF embeddings for a single chain. Shape = [L, D]"""

    device = next(model.parameters()).device

    structure = load_structure_from_gz(pdb_path, chain_id)
    coords, _seq = extract_coords_from_structure(structure)

    # coords: [L, 3, 3]
    coords = torch.tensor(coords, dtype=torch.float32, device=device)

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]

    coords_bc, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)

    with torch.no_grad():
        encoder_out = model.encoder.forward(coords_bc, padding_mask, confidence, return_all_hiddens=False)

    rep = encoder_out["encoder_out"][0][1:-1, 0]  # strip BOS/EOS
    return rep.cpu().numpy()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    dataframe_path = "/work3/s232958/data/PPint_DB/disordered_interfaces_no_cutoff_filtered_nonredundant80_3å_5.csv.gz"
    download_dir = "/work3/s232958/data/PPint_DB/pdb_cache"
    path_to_output_embeddings = "/work3/s232958/data/PPint_DB/esmif_embeddings_full"

    name_column = "PDB_chain_name"
    sequence_column = "sequence"

    os.makedirs(path_to_output_embeddings, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Load dataframe
    print(f"Reading dataframe from: {dataframe_path}", file=sys.stderr)
    df = pd.read_csv(dataframe_path)

    # Generate chain names like "1ABC_A"
    df[name_column] = df["PDB"] + "_" + df["chainname"]

    # Keep only unique usable rows
    df = df[df[sequence_column].notna()]
    df = df.drop_duplicates(subset=[name_column, sequence_column])

    # Download PDBs
    print("Downloading PDBs...", file=sys.stderr)
    df["pdb_path"] = df["PDB"].str.lower().apply(lambda pid: download_pdb(pid, download_dir))

    # Drop missing
    df = df[df["pdb_path"].notna()]
    print(f"Remaining rows after filtering: {len(df)}", file=sys.stderr)

    # Load model
    print("Loading ESM-IF1 model...", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()
    print(f"Model loaded on device: {device}", file=sys.stderr)

    # Keep track of already computed files
    existing_files = {f[:-4] for f in os.listdir(path_to_output_embeddings) if f.endswith(".npy")}
    print(f"Found {len(existing_files)} existing embeddings.", file=sys.stderr)

    # Main loop
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = row[name_column]  # e.g., 1ABC_A
        if name in existing_files:
            continue

        pdb_path = row["pdb_path"]
        chain_id = row["chainname"]

        try:
            emb = calculate_esm_if_embeddings(model, alphabet, pdb_path, chain_id)
        except Exception as e:
            print(f"Failed for {name}: {e}", file=sys.stderr)
            continue

        out_path = os.path.join(path_to_output_embeddings, f"{name}.npy")
        np.save(out_path, emb)

    print("Done.", file=sys.stderr)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
