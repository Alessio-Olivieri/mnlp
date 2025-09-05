# %%
import sys
sys.path.append("../src")
import paths
import dataset
import train
import utils
import torch
import pickle
import evaluation

from datasets import load_from_disk

# %%
train = False

# %%
TRAIN_DATA = paths.data/"manzoni_train_tokens.csv"
dev_DATA = paths.data/"manzoni_dev_tokens.csv"
HF_DATA = paths.data/"prepared"
torch.set_float32_matmul_precision("high")   # enable TF32 matmuls on Ampere
torch.backends.cudnn.allow_tf32 = True 

# %%
import numpy as np
from collections import Counter

def check_labels(hfds_split, sample_rows=2000):
    # Concatenate labels from a subset of rows (pad to same length already handled by collator)
    n = min(sample_rows, len(hfds_split))
    cats = []
    for ex in hfds_split.select(range(n)):
        labs = np.array(ex["labels"])
        cats.append(labs)
    all_labs = np.concatenate(cats)
    visible = all_labs[all_labs != -100]
    uniq = np.unique(visible)
    print("Unique visible labels:", uniq)
    bad = [x for x in uniq if x not in (0, 1)]
    if bad:
        print("❌ Found out-of-range labels:", bad)
    else:
        print("✅ Labels look fine (only 0/1).")
    return uniq

# %%
if train:
    import importlib
    importlib.reload(dataset)
    importlib.reload(train)

    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # makes the exception point to the correct op

    results = {}
    for model_key in ["deberta", "modernbert", "bert"]:
        pairs = dataset.read_token_label_file(TRAIN_DATA)
        sents_tok, sents_lab = dataset.group_into_sentences(pairs)
        ds_full = dataset.build_hf_dataset_for_token_classification(sents_tok, sents_lab, model_key=model_key)
        split = ds_full.train_test_split(train_size=0.8, seed=69)
        train_ds = dataset.tidy(split["train"], model_key)
        val_ds   = dataset.tidy(split["test"], model_key)
        train_ds.save_to_disk(HF_DATA/f"{model_key}"/"train")
        val_ds.save_to_disk(HF_DATA/f"{model_key}"/"val")
        _ = check_labels(train_ds)  # your ModernBERT train split
        _ = check_labels(val_ds)
        print(f"\n=== Training {model_key} -> {utils.MODEL_SPECS[model_key].name} ===")
        out_dir = str(paths.chekpoints / model_key)
        results[model_key] = train.train_token_splitter(
            train_ds, val_ds,
            model_key=model_key, out_dir=out_dir,
            lr=5e-5, batch_size=8, epochs=3,
        )
    with open(paths.results/"token_class_eval.pkl", "wb") as f:
        pickle.dump(results, f)
else:
    with open(paths.results/"token_class_eval.pkl", "rb") as f:
        results = pickle.load(f)

print(results)

# %%
import pandas as pd
pd.DataFrame(results).T.sort_values("eval_f1", ascending=False)

# %%
import importlib
importlib.reload(evaluation)
best_key = max(results, key=lambda k: results[k]["eval_f1"])
model_dir = paths.chekpoints/best_key
best_trainer = evaluation.load_trainer_for_eval(model_dir, HF_DATA/best_key/"val")
val_ds = load_from_disk(HF_DATA/best_key/"val")

pred = best_trainer.predict(val_ds)  # logits + label_ids as np arrays
logits = pred.predictions
label_ids = pred.label_ids
tok = best_trainer.tokenizer

def sentences_from_word_seq(words, y_pred):
    sents, cur = [], []
    for w, b in zip(words, y_pred):
        cur.append(w)
        if b == 1:
            sents.append(cur); cur = []
    if cur: sents.append(cur)
    return sents

for i in range(min(3, len(val_ds))):
    ids = val_ds[i]["input_ids"]
    words = tok.convert_ids_to_tokens(ids)

    mask = (label_ids[i] != -100)          # np.bool_ array
    y_pred = logits[i].argmax(-1)[mask]    # predicted boundary labels at visible positions
    visible_words = [w for w, m in zip(words, mask.tolist()) if m]

    sents = sentences_from_word_seq(visible_words, y_pred)
    print(f"\nWindow {i} — predicted {len(sents)} sentences:")
    print(" | ".join([" ".join(s) for s in sents]))


evaluation.preview_predictions(best_trainer, val_ds, k=3)
evaluation.preview_full_sentences(best_trainer, val_ds, n_examples=2)
evaluation.preview_pred_vs_gold(best_trainer, val_ds, [1,2,3])

# %%
def error_examples(trainer, ds, max_show=10):
    out = trainer.predict(ds)
    preds = out.predictions.argmax(-1)
    labels = out.label_ids
    mask = labels != -100
    ids = ds["input_ids"]
    tok = trainer.tokenizer
    shown = 0
    results = []
    for i in range(len(ds)):
        m = mask[i]
        if not m.any(): continue
        y_true = labels[i][m]
        y_pred = preds[i][m]
        if (y_true != y_pred).any():
            results.append(i)
            words = tok.convert_ids_to_tokens(ds[i]["input_ids"])
            visible_words = [w for w,mm in zip(words, m) if mm]
            # mark predicted boundaries with "▌"
            pieces = []
            for w, b, t in zip(visible_words, y_pred, y_true):
                mark = "▌" if b==1 else ""
                pieces.append(w+mark)
            print(" ".join(pieces))
            shown += 1
            if shown >= max_show: break
    return results

results = error_examples(best_trainer, val_ds, max_show=5)
evaluation.preview_pred_vs_gold(best_trainer, val_ds, results)



# %%
# dev
dev_results = {}
importlib.reload(dataset)

# -- dev evaluation loop:
dev_results = {}
for model_key in ["deberta", "modernbert", "bert"]:
    model_dir = paths.chekpoints / model_key
    trainer = evaluation.load_trainer_for_eval(model_dir, HF_DATA / model_key / "dev")
    pred = trainer.predict(trainer.eval_dataset)  # use their own dev eval set
    logits = pred.predictions
    labels = pred.label_ids

    metrics = evaluation.compute_prf(logits, labels)
    dev_results[model_key] = metrics
    print(f"{model_key} dev Results:", metrics)

# %%
pd.DataFrame(dev_results).T.sort_values("f1", ascending=False)

# %%
best_key = max(dev_results, key=lambda k: dev_results[k]["f1"])
model_dir = paths.chekpoints/best_key

dev_best_trainer = evaluation.load_trainer_for_eval(model_dir, HF_DATA / best_key / "dev")
dev_ds = load_from_disk(HF_DATA/best_key/"val")
results = error_examples(dev_best_trainer, dev_ds, max_show=5)

evaluation.preview_predictions(dev_best_trainer, dev_ds, k=3)
evaluation.preview_full_sentences(dev_best_trainer, dev_ds, n_examples=2)
evaluation.preview_pred_vs_gold(dev_best_trainer, dev_ds, results)


