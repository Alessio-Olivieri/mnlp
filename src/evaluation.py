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


def predict_logits(trainer, ds):
    out = trainer.predict(ds)
    return out.predictions, out.label_ids

def boundaries_from_logits(logits, labels):
    preds = logits.argmax(-1)
    mask = labels != -100
    y_true = labels[mask].ravel()     # 0/1 at first-subword positions
    y_pred = preds[mask].ravel()
    return y_true, y_pred

def detok_ids_to_words(tokenizer, input_ids):
    # Decode without special tokens; for a quick glance only
    return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

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

def pretty_print_predictions(
    trainer: Trainer,
    dataset,                # the same HF Dataset you evaluated on
    n_windows: int = 3      # how many windows to show
):
    tok = trainer.tokenizer
    out = trainer.predict(dataset)
    logits = out.predictions
    label_ids = out.label_ids

    for i in range(min(n_windows, len(dataset))):
        ids       = dataset[i]["input_ids"]
        word_mask = (dataset[i]["labels"] != -100)          # True at first sub-word of every word
        preds     = logits[i].argmax(-1)[word_mask]         # 0/1 predictions at word positions
        visible_ids = [tid for tid, m in zip(ids, word_mask) if m]

        # group token-ids into sentences
        cur_ids: List[int] = []
        sentences: List[str] = []
        for tid, boundary in zip(visible_ids, preds):
            cur_ids.append(tid)
            if boundary == 1:
                sentences.append(
                    tok.decode(cur_ids,
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True).strip()
                )
                cur_ids = []
        if cur_ids:  # last sentence if model missed final boundary
            sentences.append(tok.decode(cur_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True).strip())

        print(f"\n🪟 Window {i} – {len(sentences)} predicted sentence(s):")
        for s in sentences:
            print(" •", s)

import numpy as np

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



def _ensure_pad(tok):
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})

def parse_digits(text: str, n: int) -> List[int]:
    # grab 0/1 anywhere in the output; keep first n
    digs = [int(x) for x in re.findall(r"[01]", text)]
    if len(digs) < n:
        digs += [0] * (n - len(digs))
    return digs[:n]

def evaluate_generative_boundary_model(
    model,
    tok,
    ds = None,
    batch_size: int = 1,
    max_new_tokens_per_item: int | None = None,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
):
    """
    Runs greedy generation on ds['prompt'] and computes token-wise Acc/Precision/Recall/F1.
    Expects ds columns: prompt (str), labels (List[int]), n (int).
    """
    model.eval()
    _ensure_pad(tok)

    acc_n = prec_n = rec_n = f1_n = 0.0
    TP = FP = FN = TN = 0

    # small batching (text lengths vary; simplest is per-item loop)
    for ex in ds:
        prompt = ex["prompt"]
        gold = ex["labels"]
        n = ex["n"]

        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)

        max_new = max_new_tokens_per_item or (2 * n + 20)  # room for spaces/newlines
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0, input_ids.shape[1]:]  # only the new tokens
        text = tok.decode(gen, skip_special_tokens=True)

        pred = parse_digits(text, n)

        # metrics
        for y, p in zip(gold, pred):
            if y == 1 and p == 1: TP += 1
            elif y == 0 and p == 1: FP += 1
            elif y == 1 and p == 0: FN += 1
            else: TN += 1

    total = TP + FP + FN + TN
    acc = (TP + TN) / total if total else 0.0
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "counts": {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "total": total},
    }
