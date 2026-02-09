import hashlib
import sys
import pytorch_lightning as pl
import rootutils
import json

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
from graphany.utils import logger, timer
from graphany.utils.experiment import init_experiment
from graphany.data import GraphDataset, CombinedDataset
from graphany.model import GraphAny

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import torchmetrics
from rich.pretty import pretty_repr
import os
from datetime import datetime

mean = lambda input: np.round(np.mean(input).item(), 2)


class InductiveNodeClassification(pl.LightningModule):
    def __init__(self, cfg, combined_dataset, checkpoint=None):
        super().__init__()
        self.cfg = cfg
        self.output_file_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Load specified checkpoint
        self.train_ds = "_".join(list(combined_dataset.train_ds_dict.keys()))
        self.cfg.hparams = self.get_hparams_dict()
        hashed_ckpt_path, self.cfg.hparams_hash = self.get_hashed_ckpt_path()
        if self.cfg.train_from_scratch:
            os.remove(hashed_ckpt_path) if os.path.exists(hashed_ckpt_path) else None
        if checkpoint:
            # Initialize from previous checkpoint using previous graphany config
            ckpt = torch.load(checkpoint, map_location="cpu")
            logger.critical(f"Loaded checkpoint at {checkpoint}")
            self.gnn_model = GraphAny(**ckpt["graph_any_config"])
            self.load_state_dict(ckpt["state_dict"])
        # Auto-load checkpoint if all hparams match
        elif os.path.exists(hashed_ckpt_path):
            ckpt = torch.load(hashed_ckpt_path, map_location="cpu")
            logger.critical(f"Loaded checkpoint at {hashed_ckpt_path}")
            self.gnn_model = GraphAny(**ckpt["graph_any_config"])
            self.load_state_dict(ckpt["state_dict"])
            # If continuining training, use explicit checkpoint arg above, if trying to rerun an already trained model, don't bother, just eval it
            logger.info(
                "All hyperparameters match, skipping training and proceeding to evaluation.\n"
                f"Hyperparameters:\n{json.dumps(dict(self.cfg.hparams), indent=2)}"
            )
            self.cfg.total_steps = 0
        else:
            self.gnn_model = GraphAny(**cfg.graph_any)
        self.combined_dataset = combined_dataset
        self.attn_dict, self.loss_dict, self.res_dict = {}, {}, {}
        # Initialize accuracy metrics for validation and testing
        self.metrics = {}
        held_out_datasets = list(
            set(self.cfg._all_datasets) - set(self.cfg._trans_datasets)
        )  # 27 datasets in total
        self.heldout_metrics = [
            f"{setting}/{d.lower()[:4]}_{split}_acc"
            for split in ["val", "test"]
            for d in held_out_datasets
            for setting in ["trans", "ind"]
        ]
        for split in ("val", "test"):
            self.metrics[split] = {
                k: torchmetrics.Accuracy(task="multiclass", num_classes=v.num_class)
                for k, v in combined_dataset.eval_ds_dict.items()
            }
        # For final test only; per-channel and equal-weight channel (MeanAgg) metrics
        self.per_ch_metrics = {}
        for split in ("val", "test"):
            self.per_ch_metrics[split] = {}
            for pred_ch in cfg.pred_chn.split("+") + ["MeanAgg"]:
                self.per_ch_metrics[split][pred_ch] = {
                    k: torchmetrics.Accuracy(task="multiclass", num_classes=v.num_class)
                    for k, v in combined_dataset.eval_ds_dict.items()
                }
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_hparams_dict(self):
        """"""
        hparams = {
            k: self.cfg.get(k)
            for k in [
                "feat_chn",
                "pred_chn",
                "entropy",
                "attn_temp",
                "n_hidden",
                "n_mlp_layer",
                "limit_train_batches",
                "n_per_label_examples",
                "optimizer",
                "lr",
                "weight_decay",
                "total_steps",
                "train_batch_size",
                "val_test_batch_size",
                "seed",
                "prev_ckpt",
                "drop_one_dataset_per_batch",
            ]
        } | {"dataset": self.train_ds}

        return hparams

    def get_hashed_ckpt_path(self):
        """"""
        canonical = json.dumps(
            dict(self.cfg.hparams),
            sort_keys=True,
            separators=(",", ":"),  # remove whitespace
            ensure_ascii=True,
        )
        hash_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        hashed_ckpt_path = os.path.join(
            self.cfg.dirs.data_cache, "ckpts", f"{self.train_ds}_{hash_key}.pt"
        )
        os.makedirs(os.path.dirname(hashed_ckpt_path), exist_ok=True)
        return hashed_ckpt_path, hash_key

    def on_train_end(self):
        checkpoint_path = os.path.join(
            self.cfg.dirs.output,
            f"{self.cfg.hparams['dataset']}_val_acc={self.res_dict['val_acc']}.pt",
        )
        self.save_checkpoint(checkpoint_path)

        hashed_ckpt_path, _ = self.get_hashed_ckpt_path()
        self.save_checkpoint(hashed_ckpt_path)

    def save_checkpoint(self, file_path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": [
                opt.state_dict() for opt in self.trainer.optimizers
            ],
            "graph_any_config": self.cfg.graph_any,
        }
        torch.save(checkpoint, file_path)
        logger.critical(f"Checkpoint saved to {file_path}")

    def get_metric_name(self, ds_name, split):
        if ds_name.endswith("HopSign"):
            return f"hopsR/{ds_name.lower()[:4]}_{split}_acc"
        if ds_name.endswith("HopSignER"):
            return f"hopsER/{ds_name.lower()[:4]}_{split}_acc"
        if ds_name.endswith("HopSignG"):
            return f"hopsG/{ds_name.lower()[:4]}_{split}_acc"
        if ds_name in self.cfg.train_datasets:
            return f"trans/{ds_name.lower()[:4]}_{split}_acc"
        else:
            return f"ind/{ds_name.lower()[:4]}_{split}_acc"

    def configure_optimizers(self):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        else:  # AdamW
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        return optimizer

    def on_fit_start(self):
        super().on_fit_start()
        # move all datasets to the correct GPU device
        print(f"moving train and eval datasets to {self.device}")
        self.combined_dataset.to(self.device)
        self.move_metrics_to_device()

    def move_metrics_to_device(self):
        for metrics_dict in self.metrics.values():
            for metric in metrics_dict.values():
                metric.to(self.device)

    def predict(
        self,
        ds: list[GraphDataset],
        nodes: torch.Tensor,  # [N]
        input: dict[str, torch.Tensor],  # t: [N, C]
        is_training=False,
        **kwargs,
    ) -> torch.Tensor:
        # Use preprocessed distance during evaluation
        dist = (
            torch.cat([d.dist for d in ds]) if not is_training else None
        )  # [N, t(t-1)]
        dist = dist.to(nodes.device)[nodes] if dist is not None else dist

        # Get predictions and attentions from GNN model
        preds, attn, logits = self.gnn_model(
            logit_dict={c: chn_pred[nodes] for c, chn_pred in input.items()},
            dist=dist,
            **kwargs,
        )
        attn_mean = attn.mean(0).tolist()
        self.attn_dict.update(
            {
                f"Attention/{'_'.join([d.name for d in ds])}-{c}": v
                for c, v in zip(self.cfg.feat_channels, attn_mean)
            }
        )

        if kwargs.get("test", False):
            attn_filename = os.path.join(
                self.cfg.dirs.data_cache,
                f"{'+'.join(sorted(input.keys(), reverse=True))}_attn_weights",
                self.cfg.dataset + "_attn.pt",
            )
            if not os.path.exists(attn_filename):
                os.makedirs(os.path.dirname(attn_filename), exist_ok=True)
                torch.save(self.attn_dict, attn_filename)
            logger.info(f"\nSaved attention weights to {attn_filename}")

        return preds, logits

    def training_step(self, batch, batch_idx):
        if self.cfg.get("mixed_batch_training", False):
            return self.mixed_training_step(batch, batch_idx)
        else:
            return self.original_training_step(batch, batch_idx)

    def mixed_training_step(self, batch, batch_idx):
        """
        batch is in form {ds_name1 : batch_indices1, ds_name2 : batch_indices2, ...}
        """
        loss = {}
        all_ds, inputs, train_target_idx, ds_names = [], {}, [], []
        train_target_idx_offset, batches_offset = 0, 0
        dataset_stats = {}

        if len(batch) > 2 and self.cfg.get("drop_one_dataset_per_batch", False):
            # Remove one at random, hopefully to improve generalization
            batch.pop(np.random.choice(list(batch.keys())))

        for ds_name, batch_nodes in batch.items():
            ds = self.combined_dataset.train_ds_dict[ds_name]
            all_ds.append(ds)
            ds_names.append(ds_name)
            train_target_idx.append(train_target_idx_offset + batch_nodes)
            train_target_idx_offset += ds.n_nodes
            batches_offset += len(batch_nodes)
            # Batch nodes are not visible to avoid trivial solution and overfitting
            visible_nodes = list(
                set(ds.train_indices.tolist()) - set(batch_nodes.tolist())
            )
            ref_nodes = torch.tensor(visible_nodes, dtype=torch.long).to(self.device)
            ds_too_small = len(visible_nodes) < len(batch_nodes)
            if ds_too_small:
                # Visible nodes are too few, add first half of the batch to visible nodes
                ref_nodes = torch.cat((ref_nodes, batch_nodes[: len(batch_nodes) // 2]))

            input = ds.compute_channel_logits(
                ds.features, ref_nodes, sample=True, device=self.device
            )
            for k, v in input.items():
                if k not in inputs:
                    inputs[k] = []
                inputs[k].append(v)

        input = {k: torch.cat(v, dim=0) for k, v in inputs.items()}

        train_target_idx = torch.cat(train_target_idx, dim=0)
        ds_name = "_".join(ds_names)

        preds, logits = self.predict(
            ds=all_ds,
            nodes=train_target_idx,
            input=input,
            is_training=True,
        )

        # Loss
        kwargs = {}
        all_labels = torch.cat([d.label for d in all_ds], dim=0)
        loss[f"loss/{ds_name}_loss"] = self.criterion(
            preds, all_labels[train_target_idx], **kwargs
        )

        detached_loss = {k: v.detach().cpu() for k, v in loss.items()}
        avg_loss = mean(list(detached_loss.values()))
        self.loss_dict.update({"loss/avg_loss": avg_loss, **detached_loss})
        return sum(loss.values())

    def original_training_step(self, batch, batch_idx):
        """
        batch is in form {ds_name1 : batch_indices1, ds_name2 : batch_indices2, ...}
        """
        loss = {}
        for ds_name, batch_nodes in batch.items():
            ds = self.combined_dataset.train_ds_dict[ds_name]
            train_target_idx = batch_nodes
            # Batch nodes are not visible to avoid trivial solution and overfitting
            visible_nodes = list(
                set(ds.train_indices.tolist()) - set(batch_nodes.tolist())
            )
            ref_nodes = torch.tensor(visible_nodes, dtype=torch.long).to(self.device)
            ds_too_small = len(visible_nodes) < len(batch_nodes)
            if ds_too_small:
                # Visible nodes are too few, add first half of the batch to visible nodes
                ref_nodes = torch.cat((ref_nodes, batch_nodes[: len(batch_nodes) // 2]))

            input = ds.compute_channel_logits(
                ds.features, ref_nodes, sample=True, device=self.device
            )

            preds, logits = self.predict(
                [ds], train_target_idx, input, is_training=True
            )
            kwargs = {}
            loss[f"loss/{ds_name}_loss"] = self.criterion(
                preds, ds.label[train_target_idx], **kwargs
            )

        detached_loss = {k: v.detach().cpu() for k, v in loss.items()}
        avg_loss = mean(list(detached_loss.values()))
        self.loss_dict.update({"loss/avg_loss": avg_loss, **detached_loss})
        return sum(loss.values())

    def evaluation_step(self, split, batch, batch_idx):
        self.move_metrics_to_device()

        for ds_name, eval_idx in batch.items():
            if eval_idx is None:  # Skip if dataset is already evaluated (empty batch)
                continue
            ds = self.combined_dataset.eval_ds_dict[ds_name]

            ds.to(self.device)
            eval_idx.to(self.device)
            # Use unmasked feature for evaluation
            processed_feat = ds.unmasked_pred

            kwargs = {}
            if split == "test":
                kwargs["one_hot_test_attn"] = self.cfg.one_hot_test_attn
                kwargs["test"] = True

            preds = self.predict(
                ds=[ds],
                nodes=eval_idx,
                input=processed_feat,
                is_training=False,
                **kwargs,
            )[0].argmax(-1)
            self.metrics[split][ds_name].update(preds, ds.label[eval_idx])

            mean_agg = []
            for lin_gnn_chn, lin_gnn_preds in processed_feat.items():
                if lin_gnn_chn not in self.per_ch_metrics[split]:
                    continue  # Only evaluate channels that are in pred_chn
                self.per_ch_metrics[split][lin_gnn_chn][ds_name].update(
                    lin_gnn_preds[eval_idx].argmax(-1), ds.label[eval_idx]
                )
                mean_agg.append(lin_gnn_preds[eval_idx].unsqueeze(0))
            mean_agg = torch.mean(torch.cat(mean_agg, dim=0), dim=0)
            self.per_ch_metrics[split]["MeanAgg"][ds_name].update(
                mean_agg.argmax(-1), ds.label[eval_idx]
            )

    def validation_step(self, batch, batch_idx):
        self.evaluation_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.evaluation_step("test", batch, batch_idx)

    def compute_and_log_metrics(self, split):
        # Compute metrics from collected outputs
        res = {}
        for ds_name, metric in self.metrics[split].items():
            metric_name = self.get_metric_name(ds_name, split)
            accuracy = metric.compute().cpu().numpy()
            res[metric_name] = np.round(accuracy * 100, 2)
            metric.reset()  # Reset metrics for the next epoch

        combined_res = {f"{split}_acc": np.round(sum(res.values()) / len(res), 2)}
        combined_res[f"heldout_{split}_acc"] = mean(
            [v for k, v in res.items() if k in self.heldout_metrics]
        )
        for prefix in ["trans", "ind", "hopsER", "hopsR", "hopsG"]:
            combined_res[f"{prefix}_{split}_acc"] = mean(
                [v for k, v in res.items() if k.startswith(prefix)]
            )

        # Log per-channel and MeanAgg metrics
        for pred_ch, metrics_dict in self.per_ch_metrics[split].items():
            for ds_name, metric in metrics_dict.items():
                metric_name = f"per_ch/{self.get_metric_name(ds_name, split)}_{pred_ch}"
                accuracy = metric.compute().cpu().numpy()
                res[metric_name] = np.round(accuracy * 100, 2)
                metric.reset()  # Reset metrics for the next epoch

        # Aggregated (not just per-dataset MeanAgg stats)
        meanagg_agg = {"ind": [], "trans": [], "all": []}
        for k, v in res.items():
            if "_MeanAgg" in k:
                if "ind/" in k:
                    meanagg_agg["ind"].append(v)
                elif "trans/" in k:
                    meanagg_agg["trans"].append(v)
                meanagg_agg["all"].append(v)
        combined_res[f"ind_{split}_MeanAgg_acc"] = mean(meanagg_agg["ind"])
        combined_res[f"trans_{split}_MeanAgg_acc"] = mean(meanagg_agg["trans"])
        combined_res[f"{split}_MeanAgg_acc"] = mean(meanagg_agg["all"])

        res.update(self.attn_dict)

        self.log_dict(res, prog_bar=False, logger=True, add_dataloader_idx=False)
        self.log_dict(
            combined_res, prog_bar=True, logger=True, add_dataloader_idx=False
        )
        self.res_dict.update({**res, **combined_res})

    def on_train_epoch_end(self):
        self.log_dict(self.loss_dict, on_epoch=True, prog_bar=True, logger=True)
        if len(self.attn_dict):
            self.log_dict(self.attn_dict, on_epoch=True, prog_bar=False, logger=True)

    def on_validation_epoch_end(self):
        self.compute_and_log_metrics("val")

    def on_test_epoch_end(self):
        self.compute_and_log_metrics("test")


def infer_datasets(dataset_name):
    """Dynamically infer train and eval datasets from dataset_name string."""
    train, eval = dataset_name.split("_X_")
    train_datasets = train.split("_")
    if eval == "4HopGrid":
        eval_datasets = [
            "1HopSignG",
            "2HopSignG",
            "3HopSignG",
            "4HopSignG",
            # "5HopSignG",
            # "6HopSignG",
            "Cora",
            "FCora",
            "Citeseer",
            "DBLP",
            "Pubmed",
            "Wiki",
            "WkCS",
            "AmzComp",
            "AmzPhoto",
            "BlogCatalog",
            "CoCS",
            "CoPhysics",
            "Cornell",
            "Texas",
            "Wisconsin",
            "AirBrazil",
            "AirUS",
            "AirEU",
            "Chameleon",
            "Actor",
            "Squirrel",
            "Roman",
            "AmzRatings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ]
    else:
        eval_datasets = eval.split("_")
    return train_datasets, eval_datasets


@timer()
@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    cfg, logger = init_experiment(cfg)
    # Define the default step metric for all metrics
    # wandb.define_metric("*", step_metric="epoch")
    if torch.cuda.is_available() and cfg.preprocess_device == "gpu":
        preprocess_device = torch.device("cuda")
    else:
        preprocess_device = torch.device("cpu")

    try:
        cfg.train_datasets, cfg.eval_datasets = infer_datasets(cfg.dataset)
    except:
        pass  # use explicitly specified datasets

    def construct_ds_dict(datasets):
        datasets = [datasets] if isinstance(datasets, str) else datasets
        ds_dict = {
            dataset: GraphDataset(
                cfg,
                dataset,
                cfg.dirs.data_cache,
                cfg.train_batch_size,
                cfg.val_test_batch_size,
                preprocess_device,
            )
            for dataset in datasets
        }
        return ds_dict

    train_ds_dict = construct_ds_dict(cfg.train_datasets)
    eval_ds_dict = construct_ds_dict(cfg.eval_datasets)

    if cfg.skip_training_eval:
        logger.warning("Ending early since we just want ranges")
        sys.exit(0)  # TEMP EXIT

    combined_dataset = CombinedDataset(train_ds_dict, eval_ds_dict, cfg)

    model = InductiveNodeClassification(cfg, combined_dataset, cfg.get("prev_ckpt"))
    # Set up the checkpoint callback to save only at the end of training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.dirs.output,  # specify where to save
        filename="final_checkpoint.pt",  # set a filename
        save_top_k=0,  # do not save based on metric, just save last
        save_last=True,  # ensures only the last checkpoint is kept
        save_on_train_epoch_end=True,  # save at the end of training epoch
    )
    trainer = pl.Trainer(
        max_epochs=cfg.total_steps,
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg.limit_train_batches,
        check_val_every_n_epoch=cfg.eval_freq,
        logger=logger,
        accelerator=(
            "gpu"
            if (torch.cuda.is_available() and cfg.gpus > 0 and not cfg.train_on_cpu)
            else "cpu"
        ),
        default_root_dir=cfg.dirs.lightning_root,
    )
    dataloaders = {
        "train": combined_dataset.train_dataloader(),
        "val": combined_dataset.val_dataloader(),
        "test": combined_dataset.test_dataloader(),
    }
    if cfg.total_steps > 0:
        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
    trainer.validate(model, dataloaders=dataloaders["val"])
    trainer.test(model, dataloaders=dataloaders["test"])
    final_results = model.res_dict

    # Save final results somewhere
    final_results_path = os.path.join(
        cfg.dirs.data_cache,
        cfg.get("results_output", "results"),
        cfg.dataset,
        f"{cfg.dataset}_feat={cfg.feat_chn}_pred={cfg.pred_chn}_{cfg.hparams_hash}_results.json",
    )
    if not os.path.exists(final_results_path):
        os.makedirs(os.path.dirname(final_results_path), exist_ok=True)
        with open(final_results_path, "w") as f:
            json.dump(
                final_results
                | {
                    "dataset": cfg.dataset,
                    "hparams": OmegaConf.to_container(cfg.hparams),
                    "cfg": OmegaConf.to_container(cfg),
                },
                f,
                indent=4,
            )

    logger.critical(pretty_repr(final_results))
    logger.wandb_summary_update(final_results, finish_wandb=True)
    logger.info(f"Final results saved to {final_results_path}")


if __name__ == "__main__":
    main()
