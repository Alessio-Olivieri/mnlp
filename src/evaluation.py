from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
import torch

from datasets import load_from_disk
import train
import utils

def load_trainer_for_eval(model_dir: str, val_path: str | None = None, batch_size: int = 8):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    collator = DataCollatorForTokenClassification(tok)

    # minimal args; you don't need all your training args just to evaluate
    args = TrainingArguments(
        output_dir=model_dir,                      # reuse the model dir
        per_device_eval_batch_size=batch_size,
        report_to=[],
    )

    eval_ds = load_from_disk(val_path) if val_path else None
    trainer = Trainer(
        model=model,
        tokenizer=tok,
        data_collator=collator,
        args=args,
        eval_dataset=eval_ds,                      # can be None if you only want predict()
        compute_metrics=train.compute_metrics_boundary,  # reuse your metrics fn
    )
    return trainer

def sentences_from_word_seq(words, y_pred):
    sents, cur = [], []
    for w, b in zip(words, y_pred):
        cur.append(w)
        if b == 1:  # boundary after this word
            sents.append(cur); cur = []
    if cur: sents.append(cur)
    return sents

from typing import List
import numpy as np
from datasets import load_from_disk
from transformers import Trainer

def preview_predictions(trainer, dataset, k=3):
    tok = trainer.tokenizer
    preds_out = trainer.predict(dataset)
    logits = preds_out.predictions          # [N, L, 2]
    labels = preds_out.label_ids            # [N, L]
    
    for idx in range(min(k, len(dataset))):
        mask = labels[idx] != -100
        pred_bound = logits[idx].argmax(-1)[mask]     # 0/1 per *word*
        
        input_ids = np.array(dataset[idx]["input_ids"])[mask]  # keep visible ids only
        # --- merge sub-tokens back to words ---
        # Fast way: let HuggingFace glue them:
        words = tok.convert_tokens_to_string(
            tok.convert_ids_to_tokens(input_ids)
        ).split()  # simple split OK after convert_tokens_to_string
        
        # Split sentences by predicted boundary label
        sentences, current = [], []
        for w, b in zip(words, pred_bound):
            current.append(w)
            if b == 1:
                sentences.append(" ".join(current))
                current = []
        if current:
            sentences.append(" ".join(current))
        
        print(f"\nExample {idx} — {len(sentences)} predicted sentences:")
        for s in sentences:
            print(" •", s)


def reconstruct_sentences(tokenizer, input_ids, mask_word_start, boundaries_word):
    """
    Args
    ----
    tokenizer            : the same tokenizer used for the model
    input_ids            : list[int], tokens for ONE window
    mask_word_start      : bool array, True at first sub-token of each word
    boundaries_word      : array[int] of 0/1, same length as mask_word_start==True,
                           1 means model predicts a sentence break *after* that word
    Returns
    -------
    List[str] : detokenised sentences for this window
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

    sentences, bucket, widx = [], [], 0
    for t_idx, tok in enumerate(tokens):
        bucket.append(tok)
        if mask_word_start[t_idx]:                     # we’re at first sub-token of a word
            if boundaries_word[widx] == 1:             # model says “end sentence”
                sent = tokenizer.convert_tokens_to_string(bucket).strip()
                sentences.append(sent)
                bucket = []
            widx += 1
    if bucket:                                         # any trailing tokens
        sentences.append(tokenizer.convert_tokens_to_string(bucket).strip())
    return sentences


def preview_full_sentences(trainer, dataset, n_examples=3):
    tok = trainer.tokenizer
    out = trainer.predict(dataset)
    logits, label_ids = out.predictions, out.label_ids

    for i in range(min(n_examples, len(dataset))):
        ids   = np.array(dataset[i]["input_ids"])
        mask  = (label_ids[i] != -100)                 # True at first-subtoken positions
        preds = logits[i].argmax(-1)[mask]             # 0/1 per *word start*

        sents = reconstruct_sentences(tok, ids, (label_ids[i] != -100), preds)

        print(f"\n### Example {i} — {len(sents)} predicted sentences\n")
        for s in sents:
            print(" •", s)

import numpy as np

def reconstruct_sentences(
    tokenizer,
    input_ids: np.ndarray,
    mask_word_start: np.ndarray,        # bool per token – True at first sub-token of every word
    boundaries_word: np.ndarray,        # 0/1 per *word* – boundary after this word?
):
    """
    Detokenise a window back into full sentences, given a boundary array.
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

    sentences, bucket, widx = [], [], 0
    for t_idx, tok in enumerate(tokens):
        bucket.append(tok)
        if mask_word_start[t_idx]:                # first sub-token of a word
            if boundaries_word[widx] == 1:        # model or gold says “sentence ends here”
                sentences.append(tokenizer.convert_tokens_to_string(bucket).strip())
                bucket = []
            widx += 1
    if bucket:
        sentences.append(tokenizer.convert_tokens_to_string(bucket).strip())
    return sentences


def preview_pred_vs_gold(trainer, dataset, examples):
    """
    Prints P: (predicted) and G: (gold) sentences for `n_examples` windows.
    """
    tok   = trainer.tokenizer
    pred  = trainer.predict(dataset)
    logits, label_ids = pred.predictions, pred.label_ids

    for i in examples:
        ids   = np.array(dataset[i]["input_ids"])
        mask  = (label_ids[i] != -100)                      # bool per token
        word_pred  = logits[i].argmax(-1)[mask]             # 0/1 per word
        word_gold  = label_ids[i][mask]                     # 0/1 per word (ground truth)

        sent_pred = reconstruct_sentences(tok, ids, mask, word_pred)
        sent_gold = reconstruct_sentences(tok, ids, mask, word_gold)

        print(f"\n### Window {i}")
        print("P:", " | ".join(sent_pred))
        print("G:", " | ".join(sent_gold))

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_prf(logits: np.ndarray, labels: np.ndarray):
    """
    Compute precision, recall, F1, and accuracy for the boundary=1 class,
    ignoring tokens where labels == -100.
    """
    preds = logits.argmax(axis=-1)
    mask = labels != -100

    y_true = labels[mask].ravel()
    y_pred = preds[mask].ravel()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)}


from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def collect_token_predictions(
    trainer,
    dataset,
    word_only: bool = True,
    include_special: bool = True,
) -> pd.DataFrame:
    """
    Runs trainer.predict(dataset) and returns a DataFrame with one row per token
    (or per *word* position if word_only=True, i.e., labels != -100).

    Columns:
      sample_idx, token_idx, word_idx, token_id, token, is_special, is_word_start,
      label, pred, prob_0, prob_1
    """
    pred_out = trainer.predict(dataset)
    logits = pred_out.predictions            # [N, L, 2]
    labels = pred_out.label_ids              # [N, L]
    tok = trainer.tokenizer

    rows = []
    N = len(dataset)
    for i in range(N):
        ids = np.array(dataset[i]["input_ids"])
        toks = tok.convert_ids_to_tokens(ids, skip_special_tokens=False)
        labs = np.array(labels[i])          # may contain -100
        probs = _softmax_np(np.array(logits[i]), axis=-1)   # [L,2]
        preds = probs.argmax(-1)                               # [L]

        # word start mask (first sub-token of each word)
        mask_word_start = (labs != -100)

        word_idx = -1
        for j in range(len(ids)):
            is_word_start = bool(mask_word_start[j])
            if word_only and not is_word_start:
                continue
            if is_word_start:
                word_idx += 1

            is_special = toks[j] in getattr(tok, "all_special_tokens", [])
            if not include_special and is_special:
                continue

            label_val = int(labs[j]) if labs[j] != -100 else -100
            rows.append({
                "sample_idx": i,
                "token_idx": j,
                "word_idx": (word_idx if is_word_start else None),
                "token_id": int(ids[j]),
                "token": toks[j],
                "is_special": bool(is_special),
                "is_word_start": is_word_start,
                "label": label_val,              # 0/1 or -100
                "pred": int(preds[j]),           # 0/1
                "prob_0": float(probs[j, 0]),
                "prob_1": float(probs[j, 1]),
            })

    df = pd.DataFrame(rows)
    return df

def save_token_predictions(
    trainer,
    dataset,
    out_path: str | Path,
    word_only: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Collects predictions and saves them to CSV or Parquet based on suffix.
    Returns (path, summary dict).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = collect_token_predictions(trainer, dataset, word_only=word_only)

    if out_path.suffix.lower() == ".parquet":
        # requires pyarrow or fastparquet; use CSV if you prefer no extra deps
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".jsonl":
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(out_path, index=False)

    summary = {
        "n_samples": len(dataset),
        "n_rows": len(df),
        "word_only": word_only,
        "path": str(out_path),
    }
    return str(out_path), summary
