#!/usr/bin/env python3
import uuid, sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import math
import random

from sklearn import metrics
from scipy import stats
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.set_device(0)  # 0 == "first visible" -> actually GPU 2 on the node
print(torch.cuda.get_device_name(0))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.optim import AdamW

torch.manual_seed(0)

from accelerate import Accelerator

import matplotlib.pyplot as plt
import seaborn as sns

import training_utils.dataset_utils as data_utils
import training_utils.partitioning_utils as pat_utils

# ----------------
# Reproducibility
# ----------------
SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------
# Config / params
# ----------------
memory_verbose = False
use_wandb = True
model_save_steps = 1
train_frac = 1.0
test_frac = 1.0

embedding_dimension = 1280  # | 960 | 1152
number_of_recycles = 2
padding_value = -5000

learning_rate = 2e-5
EPOCHS = 15

# ----------------
# Helpers
# ----------------
def create_key_padding_mask(embeddings, padding_value=-5000, offset=10):
    # embeddings: (batch, seq_len, feat)
    # True where positions are padding
    return (embeddings < (padding_value + offset)).all(dim=-1)

def create_mean_of_non_masked(embeddings, padding_mask):
    # embeddings: (batch, seq_len, feat)
    # padding_mask: (batch, seq_len) True at pad positions
    seq_embeddings = []
    for i in range(embeddings.shape[0]):
        non_masked_embeddings = embeddings[i][~padding_mask[i]]  # [num_real_tokens, feat]
        if len(non_masked_embeddings) == 0:
            print("You are masking all positions when creating sequence representation")
            sys.exit(1)
        mean_embedding = non_masked_embeddings.mean(dim=0)  # [feat]
        seq_embeddings.append(mean_embedding)
    return torch.stack(seq_embeddings)

class MiniCLIP_w_transformer_crossattn(pl.LightningModule):
    def __init__(self, padding_value=-5000, embed_dimension=embedding_dimension, num_recycles=2):
        super().__init__()
        self.num_recycles = num_recycles  # recycling passes
        self.padding_value = padding_value
        self.embed_dimension = embed_dimension

        # CLIP-like logit scale
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

        # shared encoder layer
        self.transformerencoder = nn.TransformerEncoderLayer(
            d_model=self.embed_dimension,
            nhead=8,
            dropout=0.1,
            batch_first=True,
            dim_feedforward=self.embed_dimension
        )

        self.norm = nn.LayerNorm(self.embed_dimension)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dimension,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.prot_embedder = nn.Sequential(
            nn.Linear(self.embed_dimension, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
        )

    def forward(self, pep_input, prot_input,
                label=None, pep_int_mask=None, prot_int_mask=None,
                int_prob=None, mem_save=True):

        pep_mask = create_key_padding_mask(embeddings=pep_input, padding_value=self.padding_value)
        prot_mask = create_key_padding_mask(embeddings=prot_input, padding_value=self.padding_value)

        # residual states
        pep_emb = pep_input.clone()
        prot_emb = prot_input.clone()

        for _ in range(self.num_recycles):
            # self-attention encoding
            pep_trans = self.transformerencoder(self.norm(pep_emb), src_key_padding_mask=pep_mask)
            prot_trans = self.transformerencoder(self.norm(prot_emb), src_key_padding_mask=prot_mask)

            # cross-attention
            pep_cross, _ = self.cross_attn(
                query=self.norm(pep_trans),
                key=self.norm(prot_trans),
                value=self.norm(prot_trans),
                key_padding_mask=prot_mask
            )
            prot_cross, _ = self.cross_attn(
                query=self.norm(prot_trans),
                key=self.norm(pep_trans),
                value=self.norm(pep_trans),
                key_padding_mask=pep_mask
            )

            # residual update
            pep_emb = pep_emb + pep_cross
            prot_emb = prot_emb + prot_cross

        pep_seq_coding = create_mean_of_non_masked(pep_emb, pep_mask)
        prot_seq_coding = create_mean_of_non_masked(prot_emb, prot_mask)

        pep_seq_coding = F.normalize(self.prot_embedder(pep_seq_coding))
        prot_seq_coding = F.normalize(self.prot_embedder(prot_seq_coding))

        if mem_save:
            torch.cuda.empty_cache()

        scale = torch.exp(self.logit_scale).clamp(max=100.0)
        logits = scale * (pep_seq_coding * prot_seq_coding).sum(dim=-1)

        return logits

    def training_step(self, batch, device):
        embedding_pep, embedding_prot = batch
        embedding_pep = embedding_pep.to(device)
        embedding_prot = embedding_prot.to(device)

        # positives
        positive_logits = self.forward(embedding_pep, embedding_prot)

        # negatives: all mismatched pairs across batch
        rows, cols = torch.triu_indices(embedding_prot.size(0), embedding_prot.size(0), offset=1)

        negative_logits = self(
            embedding_pep[rows, :, :],
            embedding_prot[cols, :, :],
            int_prob=0.0
        )

        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits, device=device)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits, device=device)
        )

        loss = (positive_loss + negative_loss) / 2

        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, device):
        embedding_pep, embedding_prot = batch
        embedding_pep = embedding_pep.to(device)
        embedding_prot = embedding_prot.to(device)

        with torch.no_grad():
            # positive logits
            positive_logits = self(embedding_pep, embedding_prot)

            positive_loss = F.binary_cross_entropy_with_logits(
                positive_logits,
                torch.ones_like(positive_logits, device=device)
            )

            # negative logits
            rows, cols = torch.triu_indices(embedding_prot.size(0), embedding_prot.size(0), offset=1)
            negative_logits = self(
                embedding_pep[rows, :, :],
                embedding_prot[cols, :, :],
                int_prob=0.0
            )

            negative_loss = F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits, device=device)
            )

            loss = (positive_loss + negative_loss) / 2

            # build full logit matrix on the SAME DEVICE AS TENSORS
            batch_device = embedding_pep.device
            bsz = embedding_pep.size(0)

            logit_matrix = torch.zeros(
                (bsz, bsz),
                device=batch_device
            )

            logit_matrix[rows, cols] = negative_logits
            logit_matrix[cols, rows] = negative_logits

            diag_indices = torch.arange(bsz, device=batch_device)
            logit_matrix[diag_indices, diag_indices] = positive_logits.squeeze()

            # retrieval metrics
            labels = torch.arange(bsz, device=batch_device)

            # we predict binder for each peptide (argmax over rows/cols depends on convention)
            peptide_predictions = logit_matrix.argmax(dim=0)  # index of best binder for each target
            peptide_ranks = logit_matrix.argsort(dim=0).diag() + 1
            peptide_mrr = (peptide_ranks).float().pow(-1).mean()

            peptide_accuracy = peptide_predictions.eq(labels).float().mean()

            k = 3
            topk_idx = logit_matrix.topk(k, dim=0).indices  # [k, bsz]
            # does correct index appear in top-k?
            peptide_topk_accuracy = torch.any(
                (topk_idx - labels.reshape(1, -1)) == 0,
                dim=0
            ).sum() / bsz

            del logit_matrix, positive_logits, negative_logits, embedding_pep, embedding_prot

            # return scalar floats (CPU) so we can average in wrapper
            return (
                loss.item(),
                peptide_accuracy.item(),
                peptide_topk_accuracy.item()
            )

    def calculate_logit_matrix(self, embedding_pep, embedding_prot):
        """
        Build the symmetric logit matrix for AUROC/AUPR eval.
        embedding_pep: [B, Lp, D]
        embedding_prot:[B, Lt, D]
        returns [B,B] scores
        """
        # assume inputs already on same device
        batch_device = embedding_pep.device
        bsz = embedding_pep.size(0)

        rows, cols = torch.triu_indices(bsz, bsz, offset=1)

        positive_logits = self(embedding_pep, embedding_prot)
        negative_logits = self(
            embedding_pep[rows, :, :],
            embedding_prot[cols, :, :],
            int_prob=0.0
        )

        logit_matrix = torch.zeros((bsz, bsz), device=batch_device)
        logit_matrix[rows, cols] = negative_logits
        logit_matrix[cols, rows] = negative_logits

        diag_indices = torch.arange(bsz, device=batch_device)
        logit_matrix[diag_indices, diag_indices] = positive_logits.squeeze()

        return logit_matrix


# ----------------
# Utility funcs
# ----------------
def print_mem_consumption():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    print("Total memory: ", t/1e9, "GB")
    print("Reserved memory: ", r/1e9, "GB")
    print("Allocated memory: ", a/1e9, "GB")
    print("Free memory: ", f/1e9, "GB")


# ----------------
# Dataset
# ----------------
class CLIP_PPint_analysis_dataset(Dataset):
    def __init__(self, dframe, tpath, bpath, embedding_dim=1280, padding_value=-5000.0):
        super().__init__()

        self.dframe = dframe.copy()
        self.max_tlen = int(self.dframe["seq_target_len"].max())
        self.max_blen = int(self.dframe["seq_binder_len"].max())
        self.encoding_tpath = tpath
        self.encoding_bpath = bpath
        self.dframe.set_index("target_binder_id", inplace=True)
        self.accessions = self.dframe.index.astype(str).tolist()
        self.name_to_row = {name: i for i, name in enumerate(self.accessions)}
        self.samples = []

        iterator = tqdm(
            self.accessions,
            position=0,
            total=len(self.accessions),
            desc="#Loading ESM2 embeddings"
        )

        for accession in iterator:
            parts = accession.split("_")

            if len(parts) < 4:
                raise ValueError(
                    f"Expected target_binder_id to have at least 4 underscore-separated parts, got {accession}"
                )

            target_id = parts[0] + "_" + parts[1]
            binder_id = parts[2] + "_" + parts[3]

            tname = f"t_{target_id}"
            bname = f"b_{binder_id}"

            tnpy_path = os.path.join(self.encoding_tpath, f"{tname}.npy")
            bnpy_path = os.path.join(self.encoding_bpath, f"{bname}.npy")

            if not os.path.exists(tnpy_path):
                raise FileNotFoundError(f"Missing target embedding file: {tnpy_path}")
            if not os.path.exists(bnpy_path):
                raise FileNotFoundError(f"Missing binder embedding file: {bnpy_path}")

            tembd = np.load(tnpy_path)
            if tembd.shape[0] < self.max_tlen:
                t_pad_len = self.max_tlen - tembd.shape[0]
                t_pad = np.full(
                    (t_pad_len, tembd.shape[1]),
                    padding_value,
                    dtype=tembd.dtype
                )
                t_final = np.concatenate([tembd, t_pad], axis=0)
            else:
                t_final = tembd[: self.max_tlen]

            bembd = np.load(bnpy_path)
            if bembd.shape[0] < self.max_blen:
                b_pad_len = self.max_blen - bembd.shape[0]
                b_pad = np.full(
                    (b_pad_len, bembd.shape[1]),
                    padding_value,
                    dtype=bembd.dtype
                )
                b_final = np.concatenate([bembd, b_pad], axis=0)
            else:
                b_final = bembd[: self.max_blen]

            b_tensor = torch.tensor(b_final, dtype=torch.float32)
            t_tensor = torch.tensor(t_final, dtype=torch.float32)

            # store (binder, target)
            self.samples.append((b_tensor, t_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_tensor, t_tensor = self.samples[idx]
        return b_tensor, t_tensor

    def _get_by_name(self, name):
        # single accession
        if isinstance(name, str):
            idx = self.name_to_row[name]
            binder_tensor, target_tensor = self.__getitem__(idx)
            return binder_tensor, target_tensor

        # list of accessions
        binder_list = []
        target_list = []
        for n in name:
            idx = self.name_to_row[n]
            b_tensor, t_tensor = self.__getitem__(idx)
            binder_list.append(b_tensor)
            target_list.append(t_tensor)

        binder_batch = torch.stack(binder_list, dim=0)   # [B, max_blen, emb_dim]
        target_batch = torch.stack(target_list, dim=0)   # [B, max_tlen, emb_dim]
        return binder_batch, target_batch


# ----------------
# Batching helper
# ----------------
def batch(iterable, n=1):
    """Yield slices of length n from a list-like iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# ----------------
# Training wrapper
# ----------------
class TrainWrapper():
    def __init__(self,
                 model,
                 training_loader,
                 validation_loader,
                 test_df,
                 test_dataset,
                 optimizer,
                 EPOCHS,
                 runID,
                 device,
                 test_indexes_for_auROC=None,
                 auROC_batch_size=18,
                 model_save_steps=False,
                 model_save_path=False,
                 v=False,
                 wandb_tracker=None):

        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.EPOCHS = EPOCHS
        self.wandb_tracker = wandb_tracker
        self.model_save_steps = model_save_steps
        self.verbose = v
        self.best_vloss = 1_000_000
        self.optimizer = optimizer
        self.runID = runID
        self.trained_model_dir = model_save_path
        self.print_frequency_loss = 1
        self.device = device

        self.test_indexes_for_auROC = test_indexes_for_auROC
        self.auROC_batch_size = auROC_batch_size
        self.test_dataset = test_dataset
        self.test_df = test_df

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_data in tqdm(self.training_loader,
                               total=len(self.training_loader),
                               desc="Running through epoch"):

            # skip batch size 1 (can't make negatives)
            if batch_data[0].size(0) == 1:
                continue

            self.optimizer.zero_grad()
            loss = self.model.training_step(batch_data, self.device)
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            running_loss += loss.item()

        return running_loss / len(self.training_loader)

    def calc_auroc_aupr_on_indexes(self, model, dataset, test_df, nondimer_indexes, batch_size=20):
        self.model.eval()
        all_TP_scores, all_FP_scores = [], []

        accessions = [test_df.loc[index].target_binder_id for index in nondimer_indexes]
        batches_local = batch(accessions, n=batch_size)

        with torch.no_grad():
            for index_batch in tqdm(
                batches_local,
                total=int(max(1, len(accessions) / batch_size)),
                desc="Calculating AUC"
            ):
                binder_emb, target_emb = dataset._get_by_name(index_batch)
                binder_emb = binder_emb.to(self.device)
                target_emb = target_emb.to(self.device)

                logit_matrix = self.model.calculate_logit_matrix(binder_emb, target_emb)

                TP_scores = logit_matrix.diag().detach().cpu().tolist()
                all_TP_scores += TP_scores

                n = logit_matrix.size(0)
                rows, cols = torch.triu_indices(n, n, offset=1)
                FP_scores = logit_matrix[rows, cols].detach().cpu().tolist()
                all_FP_scores += FP_scores

        all_score_predictions = np.array(all_TP_scores + all_FP_scores)
        all_labels = np.array([1]*len(all_TP_scores) + [0]*len(all_FP_scores))

        # ROC / PR
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_score_predictions)
        auroc = metrics.roc_auc_score(all_labels, all_score_predictions)
        aupr = metrics.average_precision_score(all_labels, all_score_predictions)

        return auroc, aupr, all_TP_scores, all_FP_scores

    def validate(self,
                 dataloader,
                 indexes_for_auc=False,
                 auROC_dataset=False,
                 auROC_df=False):

        self.model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        running_topk_accuracy = 0.0

        with torch.no_grad():
            for batch_data in tqdm(
                dataloader,
                total=len(dataloader),
                desc="First Validation run"
            ):
                if batch_data[0].size(0) == 1:
                    continue

                loss_valstep, peptide_acc, peptide_topk_acc = self.model.validation_step(
                    batch_data,
                    self.device
                )
                running_loss += loss_valstep
                running_accuracy += peptide_acc
                running_topk_accuracy += peptide_topk_acc

            val_loss = running_loss / len(dataloader)
            val_accuracy = running_accuracy / len(dataloader)
            val_topk_accuracy = running_topk_accuracy / len(dataloader)

            if indexes_for_auc:
                non_dimer_auc, non_dimer_aupr, ___, ___ = self.calc_auroc_aupr_on_indexes(
                    model=self.model,
                    dataset=auROC_dataset,
                    test_df=auROC_df,
                    nondimer_indexes=indexes_for_auc,
                    batch_size=self.auROC_batch_size
                )

                return val_loss, val_accuracy, val_topk_accuracy, non_dimer_auc, non_dimer_aupr

            else:
                return val_loss, val_accuracy, val_topk_accuracy

    def train_model(self):
        if self.verbose:
            print(f"Training model {str(self.runID)}")

        # pre-training validation
        if self.test_indexes_for_auROC:
            (
                val_loss_before,
                val_accuracy_before,
                val_topk_accuracy_before,
                val_nondimer_auc_before,
                val_nondimer_aupr_before
            ) = self.validate(
                dataloader=self.validation_loader,
                indexes_for_auc=self.test_indexes_for_auROC,
                auROC_dataset=self.test_dataset,
                auROC_df=self.test_df
            )
        else:
            (
                val_loss_before,
                val_accuracy_before,
                val_topk_accuracy_before
            ) = self.validate(
                dataloader=self.validation_loader,
                indexes_for_auc=self.test_indexes_for_auROC,
                auROC_dataset=self.test_dataset,
                auROC_df=self.test_df
            )
            val_nondimer_auc_before = None
            val_nondimer_aupr_before = None

        if self.verbose:
            msg = (
                f'Before training - Val CLIP-loss {round(val_loss_before,4)} '
                f'Accuracy: {round(val_accuracy_before,4)} '
                f'Top-K accuracy : {round(val_topk_accuracy_before,4)}'
            )
            if val_nondimer_auc_before is not None:
                msg += f' auc: {round(val_nondimer_auc_before,3)} auPR: {round(val_nondimer_aupr_before,3)}'
            print(msg)

        if self.wandb_tracker:
            metrics_to_log = {
                "Val-loss": val_loss_before,
                "Val-acc": val_accuracy_before,
                "Val-TOPK-acc": val_topk_accuracy_before,
            }
            if val_nondimer_auc_before is not None:
                metrics_to_log["Val non-dimer auc"] = val_nondimer_auc_before
                metrics_to_log["Val non-dimer auPR"] = val_nondimer_aupr_before
            self.wandb_tracker.log(metrics_to_log)

        for epoch in tqdm(range(1, self.EPOCHS + 1),
                          total=self.EPOCHS,
                          desc="Epochs"):

            train_loss = self.train_one_epoch()

            # validation AFTER each epoch
            if self.test_indexes_for_auROC:
                (
                    val_loss,
                    val_accuracy,
                    val_topk_accuracy,
                    val_nondimer_auc,
                    val_nondimer_aupr
                ) = self.validate(
                    dataloader=self.validation_loader,
                    indexes_for_auc=self.test_indexes_for_auROC,
                    auROC_dataset=self.test_dataset,
                    auROC_df=self.test_df
                )
            else:
                (
                    val_loss,
                    val_accuracy,
                    val_topk_accuracy
                ) = self.validate(
                    dataloader=self.validation_loader,
                    indexes_for_auc=self.test_indexes_for_auROC,
                    auROC_dataset=self.test_dataset,
                    auROC_df=self.test_df
                )
                val_nondimer_auc = None
                val_nondimer_aupr = None

            # save checkpoints every N epochs if requested
            if self.model_save_steps:
                if epoch % self.model_save_steps == 0:
                    check_point_folder = os.path.join(
                        self.trained_model_dir,
                        f"{str(self.runID)}_checkpoint_{str(epoch)}"
                    )
                    if self.verbose:
                        print("Saving model to:", check_point_folder)

                    if not os.path.exists(check_point_folder):
                        os.makedirs(check_point_folder)

                    checkpoint_path = os.path.join(
                        check_point_folder,
                        f"{str(self.runID)}_checkpoint_epoch_{str(epoch)}.pth"
                    )
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss
                        },
                        checkpoint_path
                    )

            if self.verbose and (epoch % self.print_frequency_loss == 0):
                msg = (
                    f'EPOCH {epoch} -  Val loss {round(val_loss,4)} '
                    f'Accuracy: {round(val_accuracy,4)} '
                    f'Top-K accuracy: {round(val_topk_accuracy,4)}'
                )
                if val_nondimer_auc is not None:
                    msg += (
                        f' Val-Auc:{round(val_nondimer_auc,3)}'
                        f' Val-auPR:{round(val_nondimer_aupr,3)}'
                    )
                print(msg)

            if self.wandb_tracker:
                metrics_to_log = {
                    "Epoch": epoch,
                    "Train-loss": train_loss,
                    "Val-loss": val_loss,
                    "Val-acc": val_accuracy,
                    "Val-TOPK-acc": val_topk_accuracy,
                }
                if val_nondimer_auc is not None:
                    metrics_to_log["Val non-dimer auc"] = val_nondimer_auc
                    metrics_to_log["Val non-dimer auPR"] = val_nondimer_aupr

                self.wandb_tracker.log(metrics_to_log)

        if self.wandb_tracker:
            self.wandb_tracker.finish()


# ----------------
# Main script body
# ----------------
if __name__ == "__main__":

    # Paths
    trained_model_dir = "/zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/ona_drafts"
    binders_embeddings = "/zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/data/PPint_DB/binders_embeddings_esm2"
    targets_embeddings = "/zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/data/PPint_DB/targets_embeddings_esm2"

    # Load PPint dataframe
    PPint_interaactions_df = pd.read_csv(
        "/zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/data/PPint_DB/PPint_interactions.csv"
    )
    PPint_interaactions_df["seq_target_len"] = [
        len(seq) for seq in PPint_interaactions_df["seq_target"].tolist()
    ]
    PPint_interaactions_df["seq_binder_len"] = [
        len(seq) for seq in PPint_interaactions_df["seq_binder"].tolist()
    ]

    # basic split
    Df_val = PPint_interaactions_df.sample(
        n=round(len(PPint_interaactions_df) * 0.2),
        random_state=0
    )
    Df_train = PPint_interaactions_df.drop(Df_val.index)

    # build datasets
    training_Dataset = CLIP_PPint_analysis_dataset(
        Df_train,
        tpath=targets_embeddings,
        bpath=binders_embeddings,
        embedding_dim=embedding_dimension
    )
    validation_Dataset = CLIP_PPint_analysis_dataset(
        Df_val,
        tpath=targets_embeddings,
        bpath=binders_embeddings,
        embedding_dim=embedding_dimension
    )

    model = MiniCLIP_w_transformer_crossattn(
        embed_dimension=embedding_dimension,
        num_recycles=number_of_recycles
    ).to("cuda")

    # training setup
    batch_size = 20
    learning_rate = 2e-5
    g = torch.Generator().manual_seed(SEED)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(training_Dataset, batch_size=20)
    val_dataloader = DataLoader(validation_Dataset, batch_size=20, shuffle=False, drop_last=False)

    # accelerator for device handling
    accelerator = Accelerator()
    device = accelerator.device
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader
    )

    if use_wandb:
        import wandb
        run = wandb.init(
            project="PPint_retrain_w_10percent_ofdata",
            name=f"PPint_retrain",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": EPOCHS,
                "architecture": "MiniCLIP_w_transformer_crossattn",
                "dataset": "Meta analysis"
            },
        )
        wandb.watch(accelerator.unwrap_model(model), log="all", log_freq=100)
        wandb_tracker_obj = wandb
    else:
        wandb_tracker_obj = None

    # indices for AUROC on validation set
    indices_non_dimers_val = None

    runID = uuid.uuid4()

    training_wrapper = TrainWrapper(
        model=model,
        training_loader=train_dataloader,
        validation_loader=val_dataloader,
        test_dataset=validation_Dataset,
        test_df=Df_val,
        optimizer=optimizer,
        EPOCHS=EPOCHS,
        runID=runID,
        device=device,
        test_indexes_for_auROC=indices_non_dimers_val,  # can be None
        model_save_steps=model_save_steps,
        model_save_path=trained_model_dir,
        v=True,
        wandb_tracker=wandb_tracker_obj
    )

    training_wrapper.train_model()
