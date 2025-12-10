#!/usr/bin/env python3
import uuid
import sys
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator

from retraining_utils.training_utils import TrainWrapper
from retraining_utils.model_architecture import MiniCLIP_w_transformer_crossattn
from retraining_utils.datasets_utils import (
    CLIP_PPint_class, 
    CLIP_Meta_class
)


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


SEED = 0
set_seed(SEED)


# -------------------------
# Hyperparameters
# -------------------------
use_wandb = True
learning_rate = 2e-5
EPOCHS = 12
embedding_dimension = 512
number_of_recycles = 2
batch_size = 10
padding_value = -5000
model_save_steps = 3


# -------------------------
# Paths
# -------------------------
PPint_emb_path = "/work3/s232958/data/PPint_DB/esmif_embeddings_noncanonical"
Meta_bemb_path = "/work3/s232958/data/meta_analysis/esmif_embeddings_binders"
Meta_temb_path = "/work3/s232958/data/meta_analysis/esmif_embeddings_targets"

PPint_train_path = "/work3/s232958/data/PPint_DB/PPint_train.csv"
PPint_test_path = "/work3/s232958/data/PPint_DB/PPint_test.csv"

meta_csv_path = "/work3/s232958/data/meta_analysis/interaction_df_metaanal_w_pbd_lens.csv"

trained_model_dir = "/work3/s232958/trained/original_architecture/"


# ==========================================================
#                     MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":

    # -------------------------
    # Load main datasets
    # -------------------------
    Df_train = pd.read_csv(PPint_train_path, index_col=0).reset_index(drop=True)
    Df_test = pd.read_csv(PPint_test_path, index_col=0).reset_index(drop=True)

    # -------------------------
    # Load meta-analysis dataframe
    # -------------------------
    meta_df = (
        pd.read_csv(meta_csv_path)
        .drop(columns=["binder_id", "target_id"])
        .rename(columns={
            "target_id_mod": "target_id",
            "target_binder_ID": "binder_id",
        })
    )

    # Shuffle for training purposes
    interaction_df_shuffled = meta_df.sample(frac=1, random_state=SEED).reset_index(drop=True)


    # -------------------------
    # Create datasets
    # -------------------------
    training_Dataset = dataset.CLIP_PPint_dataclass(
        Df_train,
        path=PPint_emb_path,
        embedding_dim=embedding_dimension
    )

    testing_Dataset = dataset.CLIP_PPint_dataclass(
        Df_test,
        path=PPint_emb_path,
        embedding_dim=embedding_dimension
    )

    validation_Dataset = dataset.CLIP_PPint_metaanal(
        interaction_df_shuffled,
        paths=[Meta_bemb_path, Meta_temb_path],
        embedding_dim=embedding_dimension
    )


    # -------------------------
    # Create dataloaders
    # -------------------------
    train_dataloader = DataLoader(
        training_Dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        testing_Dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_dataloader = DataLoader(
        validation_Dataset,
        batch_size=20,
        shuffle=False
    )


    # -------------------------
    # Create model & optimizer
    # -------------------------
    model = model_arch.MiniCLIP_w_transformer_crossattn(
        embed_dimension=embedding_dimension,
        num_recycles=number_of_recycles
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)


    # -------------------------
    # Prepare with Accelerate
    # -------------------------
    accelerator = Accelerator()
    device = accelerator.device

    model, optimizer, train_dataloader, test_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        val_dataloader,
    )


    # -------------------------
    # Setup wandb
    # -------------------------
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="CLIP_retrain_w_PPint0.1",
            name=f"Retrain_PPint0.1_ESM2_{runID}"
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": EPOCHS,
                "architecture": "MiniCLIP_w_transformer_crossattn",
                "dataset": "Meta analysis",
            },
        )
        wandb.watch(accelerator.unwrap_model(model), log="all", log_freq=100)
        wandb_tracker_obj = wandb
    else:
        wandb_tracker_obj = None


    # -------------------------
    # Create training wrapper
    # -------------------------
    runID = uuid.uuid4()

    training_wrapper = train.TrainWrapper(
        model=model,
        training_loader=train_dataloader,
        validation_loader=val_dataloader,
        test_dataset=validation_Dataset,
        test_df=interaction_df_shuffled,  # for AUROC if needed
        optimizer=optimizer,
        EPOCHS=EPOCHS,
        runID=runID,
        device=device,
        model_save_steps=3,
        model_save_path=trained_model_dir,
        test_indexes_for_auROC=None,
        v=True,
        wandb_tracker=wandb_tracker_obj,
    )


    # -------------------------
    # Train
    # -------------------------
    training_wrapper.train_model()