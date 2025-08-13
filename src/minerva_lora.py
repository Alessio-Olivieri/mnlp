import argparse, os, math, json, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---- Reuse your code ---------------------------------------------------------
from dataset import read_token_label_file, group_into_sentences
from gptoss_sent_split import (
    SPECIAL_MARKER,          # "<BOS>"
    SYSTEM_PROMPT,           # prompt with rules
)
# if you prefer using your detok_with_offsets, import from gptoss_sent_split:
from gptoss_sent_split import detok_with_offsets

# ------------------------------------------------------------------------------
# Data prep: make prompt-completion examples from token/label CSV
# ------------------------------------------------------------------------------

def _insert_markers_by_offsets(text: str, starts: List[int], sentence_start_token_ids: List[int]) -> str:
    """Insert SPECIAL_MARKER into `text` at char positions given by token start offsets."""
    # Example: starts=[0, 5, 9, ...]; sentence_start_token_ids like [0, 7, 13]
    inserts = sorted({starts[i] for i in sentence_start_token_ids})
    out, last, delta = [], 0, 0
    for pos in inserts:
        out.append(text[last:pos + delta])
        out.append(SPECIAL_MARKER)
        last = pos + delta
        delta += len(SPECIAL_MARKER)
    out.append(text[last:])
    return "".join(out)

def _detok_sentence(tokens: List[str]) -> Tuple[str, List[int]]:
    """Detokenize a *list of tokens* while recording char starts (your logic)."""
    return detok_with_offsets(tokens)

def build_examples_from_pairs(
    pairs,
    n_sentences: int = 8,
    overlap_sentences: int = 2,
    shuffle: bool = True,
) -> List[Dict[str, str]]:
    """
    Turn a token/label CSV into prompt-completion training pairs.
    Each example is ~n_sentences long (with overlap). Completion is the text with <BOS> before each sentence.
    """
    s_tokens, s_labels = group_into_sentences(pairs)  # [[tok,...], ...], [[0,0,...,1], ...]

    # sliding window over sentences
    step = max(1, n_sentences - max(0, overlap_sentences))
    examples: List[Dict[str, str]] = []

    i = 0
    while i < len(s_tokens):
        chunk_sents = s_tokens[i : i + n_sentences]
        if not chunk_sents:
            break

        # flatten tokens & compute sentence start word indices
        flat_tokens: List[str] = [t for sent in chunk_sents for t in sent]
        text, starts = _detok_sentence(flat_tokens)

        # word index of the first token of each sentence in the chunk
        sent_starts = []
        acc = 0
        for sent in chunk_sents:
            sent_starts.append(acc)
            acc += len(sent)

        # gold: insert <BOS> before each sentence's FIRST non-space char
        target = _insert_markers_by_offsets(text, starts, sent_starts)

        # simple prompt-completion format (completion-only loss in TRL)
        prompt = (
            "### System\n"
            f"{SYSTEM_PROMPT}\n"
            "### User\n"
            f"{text}\n"
            "### Assistant\n"
        )
        completion = target  # we want the model to output the marked text

        examples.append({"prompt": prompt, "completion": completion})
        i += step

    if shuffle:
        random.shuffle(examples)
    return examples

def make_splits(
    examples: List[Dict[str, str]],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    n = len(examples)
    n_val = max(1, int(n * val_ratio))
    random.Random(seed).shuffle(examples)
    val = examples[:n_val]
    train = examples[n_val:]
    return DatasetDict(
        train=Dataset.from_list(train),
        validation=Dataset.from_list(val),
    )

# ------------------------------------------------------------------------------
# Model / Tokenizer init with (Q)LoRA
# ------------------------------------------------------------------------------

def load_tokenizer_and_model(model_id: str, qlora: bool, use_bf16: bool, attn_impl=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure our custom SPECIAL_MARKER is in the vocab to avoid fragmentation
    add = []
    if SPECIAL_MARKER not in tokenizer.get_vocab():
        add.append(SPECIAL_MARKER)
    if add:
        tokenizer.add_special_tokens({"additional_special_tokens": list({*tokenizer.additional_special_tokens, *add})})

    quant_config = None
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto" if qlora else None,  # accelerate will place devices when not quantized
        attn_implementation=attn_impl,
    )

    # If we added tokens, resize embeddings
    if add:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def lora_cfg(r=16, alpha=32, dropout=0.05) -> LoraConfig:
    # Minerva is Mistral-style; standard LoRA targets for Mistral/Llama:
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train(
    model_id: str,
    train_csv: str,
    val_csv: str | None,
    out_dir: str,
    n_sentences: int,
    overlap_sentences: int,
    epochs: float,
    lr: float,
    batch_size: int,
    grad_accum: int,
    max_seq_len: int,
    seed: int,
    qlora: bool,
    bf16: bool,
):
    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer, base_model = load_tokenizer_and_model(model_id, qlora=qlora, use_bf16=bf16)

    if qlora:
        base_model = prepare_model_for_kbit_training(base_model)

    peft_config = lora_cfg()
    # SFTTrainer will wrap the model with PEFT when peft_config is passed
    # and will compute loss only on the completion by default for prompt-completion datasets. (TRL docs)
    # response delimiter (used by the collator to mask the prompt):
    response_template = "### Assistant\n"

    # Build datasets
    train_examples = build_examples_from_csv(train_csv, n_sentences, overlap_sentences)
    if val_csv:
        val_examples = build_examples_from_csv(val_csv, n_sentences, overlap_sentences, shuffle=False)
        ds = DatasetDict(
            train=Dataset.from_list(train_examples),
            validation=Dataset.from_list(val_examples),
        )
    else:
        ds = make_splits(train_examples, val_ratio=0.05, seed=seed)

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        gradient_checkpointing=True,
        packing=True,                       # pack multiple examples per sequence
        max_seq_length=max_seq_len,
        bf16=bf16,
        dataset_num_proc=2,
        # tell SFTTrainer we're using separate prompt/completion fields:
        dataset_kwargs={"prompt_column": "prompt", "completion_column": "completion"},
        # ensure only completion contributes to loss (default for prompt-completion,
        # but set explicitly in case of version differences):
        completion_only_loss=True,
        # mask everything before this substring:
        response_template=response_template,
        model_init_kwargs={"torch_dtype": torch.bfloat16 if bf16 else torch.float16},
    )

    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=cfg,
    )

    trainer.train()
    # Save PEFT adapters
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True,
                    help="e.g., sapienzanlp/Minerva-3B-base-v1.0 or sapienzanlp/Minerva-7B-base-v1.0")
    ap.add_argument("--train_csv", type=str, required=True,
                    help="CSV with token,label per line")
    ap.add_argument("--val_csv", type=str, default=None,
                    help="Optional CSV for validation; if omitted, a small split is carved from train.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_sentences", type=int, default=8)
    ap.add_argument("--overlap_sentences", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_qlora", action="store_true", help="Disable 4-bit QLoRA and train in 16-bit PEFT instead")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(
        model_id=args.model_id,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        out_dir=args.out_dir,
        n_sentences=args.n_sentences,
        overlap_sentences=args.overlap_sentences,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        qlora=(not args.no_qlora),
        bf16=(not args.fp16),
    )

if __name__ == "__main__":
    main()