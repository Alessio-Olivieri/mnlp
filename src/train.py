# cell 1
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import datasets as hfds

import dataset
import utils


def compute_metrics_boundary(eval_pred):
    """
    Computes precision/recall/F1 for the positive class (boundary=1),
    ignoring positions with label == -100.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    y_true = labels[mask].ravel()
    y_pred = preds[mask].ravel()

    # positive class = 1 (sentence boundary)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)}

def free_after_trainer(trainer, model=None, tokenizer=None, train_ds=None, val_ds=None):
    import gc, torch
    try: trainer.callback_handler = None
    except: pass
    for obj in [trainer, model, tokenizer, train_ds, val_ds]:
        try: del obj
        except: pass
    gc.collect()
    torch.cuda.empty_cache()   # returns blocks to the allocator
    torch.cuda.ipc_collect() 

def make_trainer(
    model_key: str,
    train_ds: hfds.Dataset,
    val_ds: hfds.Dataset,
    output_dir: str,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_accum_steps: int = 1,
    fp16: bool = True, bf16 = False,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    early_stopping_patience: int = 2,
):
    model_name = utils.MODEL_SPECS[model_key].name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "O", 1: "BND"}, label2id={"O": 0, "BND": 1}
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        dataloader_pin_memory=False,               
        dataloader_num_workers=0,   
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim="paged_adamw_8bit", 
        gradient_accumulation_steps=grad_accum_steps,
        fp16=fp16, bf16=bf16,
        logging_steps=50,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_boundary,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    return trainer

# cell 5
def train_token_splitter(
    train_ds, val_ds, model_key: str, out_dir: str,
    # training knobs
    lr: float = 5e-5, batch_size: int = 8, epochs: int = 3,
    weight_decay: float = 0.01, warmup_ratio: float = 0.1,
    grad_accum_steps: int = 1,
) -> Tuple[Any, Dict[str, float]]:
    trainer = make_trainer(
        model_key=model_key,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=out_dir,
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        grad_accum_steps=grad_accum_steps,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"[{model_key}] Validation:", eval_metrics)
    trainer.save_model(out_dir)  # saves best
    free_after_trainer(trainer, trainer.model, trainer.tokenizer, train_ds, val_ds)
    return eval_metrics