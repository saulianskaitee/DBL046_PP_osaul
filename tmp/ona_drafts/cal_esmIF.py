#!/usr/bin/env python3
import os, sys
import tempfile
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import torch

from esm.inverse_folding.util import (
    load_structure,
    extract_coords_from_structure,
    get_encoder_output,
    CoordBatchConverter
)

warnings.simplefilter(action="ignore", category=FutureWarning)

modified_to_canonical = {
    # A
    '2AS': 'ASP', '3AH': 'HIS', '4BF': 'PHE', '5HP': 'GLU', '5OW': 'LYS',
    '8LJ': 'LEU',  "05O": "PRO", "6M6": "MET", "9MN": "MET", "6CW": "TRP",
    "2CO" : "CYS", "6FL" : "PHE", "2MR": "ARG", "02A": "SER", "7ID": "TYR",
    "5CT": "CYS", "4MM": "MET",
    
    'ACL': 'ARG', 'AGM': 'ARG', 'AIB': 'ALA', 'ALM': 'ALA', 'ALO': 'THR',
    'ALY': 'LYS', 'ARM': 'ARG', 'ASA': 'ASP', 'ASB': 'ASP', 'ASK': 'ASP',
    'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA', 'AZK': 'LYS', "A8E": "ALA",
    "ACA": "ALA", "AAR": "ARG", "AME" : "ALA", "API": "ILE",

    # B
    'BCS': 'CYS', 'BHD': 'ASP', 'B3A': 'ALA', 'B3E': 'GLU', 'B3K': 'LYS',
    'B3S': 'SER', 'B3X': 'TRP', 'B3Y': 'TYR', 'BE2': 'GLU', "B3M" : "MET",
    'BNN': 'ALA', 'BUG': 'LEU', "BFD": "ASP", "B3T": "THR", 'BIL' : 'ILE',
    "B3L": "LEU", "DBB": "ASN",

    # C
    'C5C': 'CYS', 'C6C': 'CYS', 'CAS': 'CYS', 'CSD': 'CYS',
    'CSO': 'CYS', 'CSP': 'CYS', 'CSS': 'CYS', 'CSU': 'CYS',
    'CSW': 'CYS', 'CSX': 'CYS', 'CY1': 'CYS', 'CY3': 'CYS',
    'CYG': 'CYS', 'CYM': 'CYS', 'CYQ': 'CYS', 'CXM': 'MET',
    'CME': 'CYS', 'CSA': 'CYS', 'CXM': 'MET', "CCS": "CYS",
    "CSK": "LYS", "CMT": "CYS",

    # D
    'DAH': 'PHE', 'DAL': 'ALA', 'DAR': 'ARG', 'DAS': 'ASP', 
    'DGL': 'GLU', 'DGN': 'GLN', 'DHA': 'ALA', 'DHI': 'HIS',
    'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'LEU', 'DLY': 'LYS',
    'DNP': 'ALA', 'DPN': 'PHE', 'DPR': 'PRO', 'DSN': 'SER',
    'DSP': 'ASP', 'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR',
    'DVA': 'VAL', 'DV7': 'ASP', "D0Q" : "ASP", 'DCY': 'CYS',
    "DAM": "ASP", "DBB": "ASN",

    # E
    'EFC': 'CYS', "EO2": "GLU",

    # F
    'FLA': 'ALA', 'FME': 'MET', "F2Y": "TYR", "FY3": "TYR",
    "FTR": "TRP",    

    # G
    'GGL': 'GLU', 'GL3': 'GLY', 'GLZ': 'GLY', 'GMA': 'GLU',
    'GSC': 'GLY',

    # H
    'HAC': 'ALA', 'HAR': 'ARG', 'HIC': 'HIS', 'HIP': 'HIS',
    'HMR': 'ARG', 'HP9': 'HIS', 'HPQ': 'PHE', 'HTR': 'TRP',
    'HYP': 'PRO', "HCS": "CYS", "HZP": "PRO",

    # I
    'IAS': 'ASP', 'IIL': 'ILE', 'IYR': 'TYR',

    # K
    'KCX': 'LYS', 'KPI': 'LYS', 'KYN': 'TRP', "KFP": "LYS",
    "KYQ" : "LYS", "KGC": "LYS", "KHB": "LYS", "KCR": "LYS", 

    # L
    'LLP': 'LYS', 'LLY': 'LYS', 'LPS': 'SER', "LCK" : "LYS",
    'LTR': 'TRP', 'LYM': 'LYS', 'LYR': 'LYS', 'LYZ': 'LYS',
    'LDH': 'LEU',

    # M
    'MAA': 'ALA', 'MEN': 'ASN',   # Your earlier MEN=LYS was incorrect; MEN is ASN-derivative
    'MHS': 'HIS', 'MIS': 'SER', 'MK8': 'LEU', 'MLE': 'LEU',
    'MLY': 'LYS', 'MLZ': 'LYS', 'MPQ': 'GLY', 'MSA': 'GLY',
    'MSE': 'MET', 'MHO': 'MET', 'MVA': 'VAL', "6M6": "MET",
    "MSO": "MET", "MLL": "LEU", "M3L": "LYS", "M0H": "MET", 

    # N
    'NEM': 'HIS', 'NEP': 'HIS', 'NLE': 'LEU', 'NLN': 'LEU',
    'NLP': 'LEU', 'NMC': 'GLY', 'NFA': 'PHE', 'SNN': 'ASN',
    "NIY": "TYR", "N7P": "PRO",

    # O
    'OAS': 'SER', 'OCS': 'CYS', 'OMT': 'MET', 'ONL': 'LYS',
    '0W6': 'TRP', "ORN": "LYS", "OCY" : "CYS", "ONH": "ASN",

    # P
    'PAQ': 'TYR', 'PCA': 'GLU', 'PEC': 'CYS', 'PHI': 'PHE',
    'PHL': 'PHE', 'PR3': 'CYS', 'PRR': 'ALA', 'PTR': 'TYR',
    'PVO': 'PRO', 'PYX': 'CYS', "PHD": "ASP", "P1L": "PRO",

    # Q
    "QPA": "PHE", "QCS": "CYS", 

    # R
    'RPI': 'ARG', "RGP": "ARG",

    # S
    'SAC': 'SER', 'SAR': 'SER', 'SCH': 'CYS', 'SCS': 'CYS',
    'SCY': 'CYS', 'SEL': 'SER', 'SEP': 'SER', 'SET': 'SER',
    'SHC': 'CYS', 'SHR': 'LYS', 'SMC': 'CYS', 'SME': 'MET',
    'SOC': 'CYS', 'STY': 'TYR', 'SVA': 'SER', 'SNC': 'CYS',
    'S2C': 'CYS', "SAH": "ALA", "SEB": "SER", "SVV": "VAL",
    

    # T
    'TIH': 'ALA', 'TPL': 'TRP', 'TPO': 'THR', 'TPQ': 'ALA',
    'TRG': 'LYS', 'TRO': 'TRP', 'TSY': 'TYR', 'TYS': 'TYR',
    'TYB': 'TYR', 'TYI': 'TYR', 'TYQ': 'TYR', 'TYY': 'TYR',
    "TRQ": "TRP", "TY2": "TYR", "TYE": "TYR", "TYC": "TYR",
    'THC' : 'THR',

    # Y
    'YCM': 'CYS',

    # V
    "VLM": "VAL",

    # X
    "X2W": "TRP",

    # Z
    'ZBZ': 'ALA', "ZIQ": "GLN",
}

# dataframe_path = "/work3/s232958/data/PPint_DB/PPint_test.csv"
# dataframe_path = "/work3/s232958/data/PPint_DB/PPint_train.csv"
dataframe_path = "/work3/s232958/data/meta_analysis/interaction_df_metaanal.csv"

# download_dir = "/work3/s232958/data/PPint_DB/pdb_cache"
download_dir  = "/work3/s232958/data/meta_analysis/input_pdbs/"

name_column = "PDB_chain_name"
sequence_column = "sequence"

# Where to save ESM-IF embeddings (.npy per chain)
# path_to_output_embeddings = "/work3/s232958/data/PPint_DB/esmif_embeddings_noncanonical"
path_to_output_embeddings = "/work3/s232958/data/meta_analysis/esmif_embeddings_noncanonical"

# Root of your local PDB mirror (with subfolders a0, a1, etc.)
# root_pdb_dir = "/work3/s232958/data/PPint_DB/pdb_cache"
root_pdb_dir = "/work3/s232958/data/meta_analysis/input_pdbs/"


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
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
    # print(f"Downloading {pdb_id} from RCSB...", file=sys.stderr)

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
        for mod, canon in modified_to_canonical.items():
            mask = structure.res_name == mod
            structure.res_name[mask] = canon
    
    return structure

def calculate_esm_if_embeddings(model, alphabet, pdb_path: str, chain_id: str, device="cuda") -> np.ndarray:
    """
    Returns per-residue ESM-IF encoder embeddings for a single chain.
    Shape: [L, D]
    """
    structure = load_structure_from_gz(pdb_path, chain_id)
    coords, _seq = extract_coords_from_structure(structure)
    coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
    
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
    with torch.no_grad():
        encoder_out = model.encoder.forward(coords, padding_mask, confidence, return_all_hiddens=False)
    # remove beginning and end (bos and eos tokens)
    return encoder_out['encoder_out'][0][1:-1, 0].cpu().numpy()


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

    print("Loading datasets...")

    if dataset == "PPint":
        sequence_df = pd.read_csv(dataframe_path)
    sequence_df["pdb_fname"] = [f"{file}.pdb.gz" for file in sequence_df.binder_id]

    download_dir = "/work3/s232958/data/PPint_DB/pdb_cache"
    os.makedirs(download_dir, exist_ok=True)
    print(f"Saving contact maps to: {out_dir}")

    # Creating dirs
    os.makedirs(path_to_output_embeddings, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    
    # Loading Df
    print(f"Reading dataframe from: {dataframe_path}", file=sys.stderr)
    sequence_df = pd.read_csv(dataframe_path)
    sequence_df["pdb_fname"] = [f"{file}.pdb.gz" for file in sequence_df.binder_id]
    pdbs_by_binderid = list(sequence_df["pdb_fname"]) 
    all_pdbs = os.listdir("/work3/s232958/data/meta_analysis/input_pdbs")
    sequence_df

    # -----------------------------------------------------
    # Load ESM2 model
    # -----------------------------------------------------
    # Load ESM-IF model
    print("Loading ESM-IF1 model...", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()
    print(f"ESM2 model loaded successfully on device: {device}", file=sys.stderr)

    # -----------------------------------------------------
    # Contact map generation
    # -----------------------------------------------------
    if dataset == "meta":

        print(f"Computing contact maps for {len(interaction_dict_B)} binders...")
        for name, seq in tqdm(interaction_dict_B.items()):
            data = [(name, seq)]
            _, batch_strs, batch_tokens = batch_converter(data)

            try:
                contact_map = compute_contact_map(model, batch_tokens, [len(batch_strs[0])])
                np.save(os.path.join(out_dir_B, f"{name}.npy"), contact_map)

            except Exception as e:
                print(f"[ERROR] Failed binder {name}: {e}", file=sys.stderr)

        print(f"Computing contact maps for {len(interaction_dict_T)} targets...")
        for name, seq in tqdm(interaction_dict_T.items()):
            data = [(name, seq)]
            _, batch_strs, batch_tokens = batch_converter(data)

            try:
                contact_map = compute_contact_map(model, batch_tokens, [len(batch_strs[0])])
                np.save(os.path.join(out_dir_T, f"{name}.npy"), contact_map)

            except Exception as e:
                print(f"[ERROR] Failed target {name}: {e}", file=sys.stderr)

    else:
        print(f"Computing contact maps for {len(interaction_dict)} sequences...")
        for name, seq in tqdm(interaction_dict.items()):
            data = [(name, seq)]
            _, batch_strs, batch_tokens = batch_converter(data)

            try:
                contact_map = compute_contact_map(model, batch_tokens, [len(batch_strs[0])])
                np.save(os.path.join(out_dir, f"{name}.npy"), contact_map)

            except Exception as e:
                print(f"[ERROR] Failed sequence {name}: {e}", file=sys.stderr)

    print("All contact maps computed and saved.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()