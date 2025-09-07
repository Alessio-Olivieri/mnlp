from typing import *
from pathlib import Path
import utils
from dataclasses import dataclass

from transformers import AutoTokenizer
import datasets as hfds

def read_token_label_file(path: str | Path) -> List[Tuple[str, int]]:
    """
    Expects lines like: token,label
    Example:
        l',0
        acqua,0
        .,1
        ...
    Returns a flat list of (token, label) where label==1 means sentence ends *after* this token.
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # allow blank lines; just skip
                continue
            # Split only on the last comma in case tokens contain commas
            if "," not in line:
                # tolerate malformed lines
                continue
            token, lbl = line.rsplit(",", 1)
            token = token.strip()
            try:
                label = int(lbl)
                label = 1 if label == 1 else 0
            except ValueError:
                continue
            pairs.append((token, label))
    return pairs


def group_into_sentences(pairs: List[Tuple[str, int]]) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Split flat (token,label) pairs into sentence-level lists.
    label==1 marks sentence end *after* that token.
    Returns:
      sentences_tokens: List of [t1, t2, ..., tk]
      sentences_labels: List of [0, 0, ..., 1] (same length as tokens)
    """
    sent_tokens, sent_labels = [], []
    sentences_tokens, sentences_labels = [], []

    for tok, lbl in pairs:
        sent_tokens.append(tok)
        sent_labels.append(lbl)
        if lbl == 1:
            sentences_tokens.append(sent_tokens)
            sentences_labels.append(sent_labels)
            sent_tokens, sent_labels = [], []

    # If trailing tokens without a closing label exist, keep them as a last sentence
    if sent_tokens:
        # ensure last token is treated as sentence end
        if sent_labels:
            sent_labels[-1] = 1
        sentences_tokens.append(sent_tokens)
        sentences_labels.append(sent_labels)

    return sentences_tokens, sentences_labels


def build_hf_dataset_for_token_classification(
    sentences_tokens: List[List[str]],
    sentences_labels: List[List[int]],
    model_key: str = "deberta",
) -> hfds.Dataset:
    assert model_key in utils.MODEL_SPECS, f"Unknown model_key: {model_key}"
    spec = utils.MODEL_SPECS[model_key]
    print(spec.name)
    tokenizer = AutoTokenizer.from_pretrained(spec.name, use_fast=True)

    # Flatten all sentences into one long stream (word-level)
    words, labels = [], []
    for toks, labs in zip(sentences_tokens, sentences_labels):
        words.extend(toks)
        labels.extend(labs)

    # Make a 1-row dataset with that stream (the map below will split it into many rows)
    raw_ds = hfds.Dataset.from_dict({"tokens": [words], "labels": [labels]})

    def tokenize_and_align_labels(examples):
        # examples["tokens"] is a list of lists (batch); here batch size = 1, but keep it generic
        enc = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=spec.max_length,
            stride=spec.stride,
            return_offsets_mapping=False,
        )

        # Map each overflowed chunk back to the original sample in this batch
        sample_mapping = enc["overflow_to_sample_mapping"]

        aligned_labels = []
        # Build labels per produced chunk
        for i in range(len(enc["input_ids"])):
            # which original example in this batch did this chunk come from?
            sample_idx = sample_mapping[i]
            # labels for that original example
            labels_for_sample = examples["labels"][sample_idx]

            word_ids = enc.word_ids(batch_index=i)
            prev_wid = None
            chunk_labels = []
            for wid in word_ids:
                if wid is None:
                    chunk_labels.append(-100)  # special tokens
                elif wid != prev_wid:
                    # label only the first subword of each word
                    # (wid indexes into the word-level labels of THIS sample)
                    chunk_labels.append(labels_for_sample[wid])
                else:
                    chunk_labels.append(-100)
                prev_wid = wid
            aligned_labels.append(chunk_labels)

        enc["labels"] = aligned_labels
        return enc

    tokenized = raw_ds.map(
        tokenize_and_align_labels,
        batched=True,                 # <— important
        remove_columns=["tokens", "labels"],
    )

    # One row per produced chunk
    tokenized = tokenized.flatten_indices()

    # Optional: ModernBERT doesn't use token_type_ids; harmless to keep, or drop them here
    # if model_key == "modernbert" and "token_type_ids" in tokenized.features:
    #     tokenized = tokenized.remove_columns("token_type_ids")

    return tokenized


def tidy(ds, model_key):
    cols = set(ds.column_names)
    drop = [c for c in ["overflow_to_sample_mapping"] if c in cols]

    # Some tokenizers don't provide token_type_ids (e.g., ModernBERT may ignore them);
    # drop if present but not expected.
    tok = AutoTokenizer.from_pretrained(utils.MODEL_SPECS[model_key].name, use_fast=True)
    if "token_type_ids" in cols and "token_type_ids" not in tok.model_input_names:
        drop.append("token_type_ids")
    return ds.remove_columns(drop) if drop else ds

def sentences_from_word_seq(words, y_pred):
    sents, cur = [], []
    for w, b in zip(words, y_pred):
        cur.append(w)
        if b == 1:  # boundary after this word
            sents.append(cur); cur = []
    if cur: sents.append(cur)
    return sents
