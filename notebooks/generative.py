# %%
# !pip install --upgrade torch accelerate kernels
# !pip install git+https://github.com/huggingface/transformers triton==3.4 git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
# !pip uninstall torchvision torchaudio -y

# %%
gpt_train_set = False
gpt_dev_set = True
mistral01 = False
mistral03 = False
qwen2 = False
classic = False


train_minerva7b = False
eval_minerva7b = False

# %%
import sys
sys.path.append('../src')
import paths
from huggingface_hub import login
#Token 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
login(token = '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')

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config
if gpt_train_set or gpt_dev_set:

    model_id = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_id)

    quantization_config=Mxfp4Config.from_dict(config.quantization_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="cuda",
    )
    model.eval()

# %%
import pandas as pd
import dataset
import importlib
importlib.reload(dataset)
import gptoss_sent_split
importlib.reload(gptoss_sent_split)
from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if gpt_dev_set:

    cfg = BOSConfig(max_new_tokens=10000, n_sentences=3, batch_size=16)

    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    y_pred, skipped_jobs = run_bos_labeling(jobs, model, tokenizer, cfg, model_id=model_id)

    tokens = [t for (t, _) in pairs]
    gold = [y for (_, y) in pairs]

    # Align lengths, just in case
    n = min(len(tokens), len(y_pred))
    tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    sents = sentences_from_word_seq(tokens, y_pred)
    import pickle
    with open(paths.results/'gptpredval.pkl', 'wb') as f:
        pickle.dump((y_pred, skipped_jobs), f)
with open(paths.results/'gptpredval.pkl', 'rb') as f:
    y_pred, skipped_jobs = pickle.load(f)

pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
gold = [y for (_, y) in pairs]
print(len(gold))
print(len(y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(
    gold, y_pred, labels=[1], average="binary", zero_division=0
)
acc = accuracy_score(gold, y_pred)
print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

# Create set of all token indices in skipped jobs
skipped_token_indices = set()
for job_idx in skipped_jobs:
    job = jobs[job_idx]
    start_token = job["start"]
    end_token = start_token + len(job["tokens"])  # All tokens in this job
    skipped_token_indices.update(range(start_token, end_token))

# Create new gold and pred lists excluding tokens from skipped jobs
new_gold = [label for idx, label in enumerate(gold) if idx not in skipped_token_indices]
new_y_pred = [pred for idx, pred in enumerate(y_pred) if idx not in skipped_token_indices]

# Second evaluation: only non-skipped tokens
prec, rec, f1, _ = precision_recall_fscore_support(
    new_gold, new_y_pred, labels=[1], average="binary", zero_division=0
)
acc = accuracy_score(new_gold, new_y_pred)
print("Non-skipped tokens only:")
print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

# %% [markdown]
# # Similar models
# 

# %%
if mistral01:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    sys.path.append('../src')
    import paths
    import pandas as pd
    import dataset
    import importlib
    importlib.reload(dataset)
    import gptoss_sent_split
    importlib.reload(gptoss_sent_split)
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq

    def remove_indices(data, indices_to_remove):
        result = [item for idx, item in enumerate(data) if idx not in indices_to_remove]
        return result

    # Choose any compatible model from above
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Example

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     torch_dtype="auto",
    #     device_map="cuda",
    # )

    # Your existing code will work the same way
    cfg = BOSConfig(max_new_tokens=512, n_sentences=3)
    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    # y_pred, skipped_jobs = run_bos_labeling(jobs, model, tokenizer, cfg)
    # tokens = [t for (t                      , _) in pairs]
    # gold = [y for (_, y) in pairs]

    # # Align lengths, just in case
    # n = min(len(tokens), len(y_pred))
    # tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    # sents = sentences_from_word_seq(tokens, y_pred)
    # import pickle
    # with open(paths.results/'Mistral-7B-Instruct-v0.1-dev.pkl', 'wb') as f:
    #     pickle.dump((y_pred, skipped_jobs), f)
    # import pickle
    with open(paths.results/'Mistral-7B-Instruct-v0.1-dev.pkl', 'rb') as f:
        y_pred, skipped_jobs = pickle.load(f)

    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    gold = [y for (_, y) in pairs]
    print(len(gold))
    print(len(y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(gold, y_pred)
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

    # Create set of all token indices in skipped jobs
    skipped_token_indices = set()
    for job_idx in skipped_jobs:
        job = jobs[job_idx]
        start_token = job["start"]
        end_token = start_token + len(job["tokens"])  # All tokens in this job
        skipped_token_indices.update(range(start_token, end_token))

    # Create new gold and pred lists excluding tokens from skipped jobs
    new_gold = [label for idx, label in enumerate(gold) if idx not in skipped_token_indices]
    new_y_pred = [pred for idx, pred in enumerate(y_pred) if idx not in skipped_token_indices]

    # Second evaluation: only non-skipped tokens
    prec, rec, f1, _ = precision_recall_fscore_support(
        new_gold, new_y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(new_gold, new_y_pred)
    print("Non-skipped tokens only:")
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

# %%
if mistral03:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    sys.path.append('../src')
    import paths
    import pandas as pd
    import dataset
    import importlib
    importlib.reload(dataset)
    import gptoss_sent_split
    importlib.reload(gptoss_sent_split)
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq

    def remove_indices(data, indices_to_remove):
        result = [item for idx, item in enumerate(data) if idx not in indices_to_remove]
        return result

    # Choose any compatible model from above
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # Example

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     torch_dtype="auto",
    #     device_map="cuda",
    # )

    # Your existing code will work the same way
    cfg = BOSConfig(max_new_tokens=1024, n_sentences=3)
    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    # y_pred, skipped_jobs = run_bos_labeling(jobs, model, tokenizer, cfg)
    # tokens = [t for (t                      , _) in pairs]
    # gold = [y for (_, y) in pairs]

    # Align lengths, just in case
    # n = min(len(tokens), len(y_pred))
    # tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    # sents = sentences_from_word_seq(tokens, y_pred)
    # import pickle
    # with open(paths.results/'Mistral-7B-Instruct-v0.3-dev.pkl', 'wb') as f:
    #     pickle.dump((y_pred, skipped_jobs), f)
    # import pickle
    with open(paths.results/'Mistral-7B-Instruct-v0.3-dev.pkl', 'rb') as f:
        y_pred, skipped_jobs = pickle.load(f)

    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    gold = [y for (_, y) in pairs]
    print(len(gold))
    print(len(y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(gold, y_pred)
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

    # Create set of all token indices in skipped jobs
    skipped_token_indices = set()
    for job_idx in skipped_jobs:
        job = jobs[job_idx]
        start_token = job["start"]
        end_token = start_token + len(job["tokens"])  # All tokens in this job
        skipped_token_indices.update(range(start_token, end_token))

    # Create new gold and pred lists excluding tokens from skipped jobs
    new_gold = [label for idx, label in enumerate(gold) if idx not in skipped_token_indices]
    new_y_pred = [pred for idx, pred in enumerate(y_pred) if idx not in skipped_token_indices]

    # Second evaluation: only non-skipped tokens
    prec, rec, f1, _ = precision_recall_fscore_support(
        new_gold, new_y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(new_gold, new_y_pred)
    print("Non-skipped tokens only:")
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

# %%
if qwen2:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    sys.path.append('../src')
    import paths
    import pandas as pd
    import dataset
    import importlib
    importlib.reload(dataset)
    import gptoss_sent_split
    importlib.reload(gptoss_sent_split)
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq

    def remove_indices(data, indices_to_remove):
        result = [item for idx, item in enumerate(data) if idx not in indices_to_remove]
        return result

    # Choose any compatible model from above
    model_id = "Qwen/Qwen2-7B-Instruct"  # Example

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda",
    )

    # Your existing code will work the same way
    cfg = BOSConfig(max_new_tokens=1024, n_sentences=3)
    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    y_pred, skipped_jobs = run_bos_labeling(jobs, model, tokenizer, cfg)
    tokens = [t for (t                      , _) in pairs]
    gold = [y for (_, y) in pairs]

    # Align lengths, just in case
    n = min(len(tokens), len(y_pred))
    tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    sents = sentences_from_word_seq(tokens, y_pred)
    import pickle
    with open(paths.results/'Qwen2-7B-Instruct-dev.pkl', 'wb') as f:
        pickle.dump((y_pred, skipped_jobs), f)
    import pickle
    with open(paths.results/'Qwen2-7B-Instruct-dev.pkl', 'rb') as f:
        y_pred, skipped_jobs = pickle.load(f)

    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    gold = [y for (_, y) in pairs]
    print(len(gold))
    print(len(y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(gold, y_pred)
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

    prec, rec, f1, _ = precision_recall_fscore_support(
        remove_indices(gold, skipped_jobs), remove_indices(y_pred, skipped_jobs), labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(gold, y_pred)
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

# %% [markdown]
# # Classic

# %%
if classic:
    # Fallback-only sentence boundary baseline (no model calls)
    import importlib, pickle
    import dataset
    importlib.reload(dataset)
    importlib.reload(gss)

    from gptoss_sent_split import read_token_label_file, _fallback_punct_labels, sentences_from_word_seq
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Load tokens/gold labels
    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    tokens = [t for (t, _) in pairs]
    gold   = [y for (_, y) in pairs]

    # Predict with the simple punctuation heuristic
    y_pred = _fallback_punct_labels(tokens)

    # (optional) save predictions for later comparison
    with open(paths.results/'punctpredval.pkl', 'wb') as f:
        pickle.dump(y_pred, f)

    # Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold, y_pred, labels=[1], average="binary", zero_division=0
    )
    acc = accuracy_score(gold, y_pred)

    print(len(gold), len(y_pred))
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)})

    # (optional) reconstruct predicted sentences
    sents = sentences_from_word_seq(tokens, y_pred)
    # 'sents' is a list of token lists; use as needed


# %% [markdown]
# # Minerva finetuning that doesnt work well

# %%
if train_minerva7b:
    from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq, SPECIAL_MARKER
    from minerva_lora import load_tokenizer_and_model

    MINERVA7B = "sapienzanlp/Minerva-7B-base-v1.0"
    bf16 = True
    tokenizer, model = load_tokenizer_and_model(MINERVA7B, qlora=True, use_bf16=bf16)

    import minerva_lora
    import importlib
    importlib.reload(minerva_lora)
    from minerva_lora import build_examples_from_pairs, make_splits, lora_cfg

    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    jobs = build_examples_from_pairs(pairs, 5, 1)
    ds = make_splits(jobs, 0.1)
    ds

    from transformers import EarlyStoppingCallback   # NEW
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig
    import torch
    import paths

    # --- Training config ---
    from transformers import EarlyStoppingCallback, TrainerCallback

    class ConsoleLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs: 
                return
            # drop the huge/boring keys
            drop = {"total_flos","train_runtime","train_samples_per_second","train_steps_per_second"}
            clean = {k: v for k, v in logs.items() if k not in drop}
            print(f"[step {state.global_step}/{state.max_steps}] {clean}")

    cfg = SFTConfig(
        output_dir=paths.chekpoints/"minerva",
        num_train_epochs=2,
        per_device_train_batch_size=5,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        # <- logging every optimizer step
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        # <- eval + early stopping
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # make stdout prints instead of only a tqdm bar:
        disable_tqdm=True,
        log_level="info",
        report_to=None,  # or "none"
        gradient_checkpointing=True,
        bf16=True,
        dataset_num_proc=2,
        dataset_kwargs={"prompt_column":"prompt","completion_column":"completion"},
        completion_only_loss=True,
    )

    peft_config = lora_cfg()

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=cfg,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0),
            ConsoleLogger(),
        ],
    )
    trainer.train()

    # Save PEFT adapters + tokenizer
    trainer.model.save_pretrained(paths.chekpoints/"minerva")
    tokenizer.save_pretrained(paths.chekpoints/"minerva")


# %%
if eval_minerva7b:
    import paths
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from gptoss_sent_split import (
        SPECIAL_MARKER,          # "<BOS>"
        SYSTEM_PROMPT,           # prompt with rules
    )

    # Paths
    checkpoint_dir = paths.chekpoints / "minerva"
    base_model_name = "sapienzanlp/Minerva-7B-base-v1.0"

    # 1. Load tokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)

    # 2. Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # 3) If sizes differ, resize embeddings to tokenizer size (adds the new row)
    if base_model.get_input_embeddings().weight.shape[0] != len(tok):
        base_model.resize_token_embeddings(len(tok))
        try:
            base_model.tie_weights()   # safe if the model ties lm_head <-> embeddings
        except Exception:
            pass

    # 3. Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval(); model.config.use_cache = True


# %%
import sys
sys.path.append('../src')

import paths
from peft import PeftModel

import minerva_lora
import importlib
importlib.reload(minerva_lora)
from minerva_lora import build_examples_from_pairs, make_splits, lora_cfg
pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
jobs = build_examples_from_pairs(pairs, 5, 1)

# inputs = tok(jobs[0]['prompt'], return_tensors="pt").to(model.device)
# out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
# gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

# %%
import re

def normalize_markers(s: str) -> str:
    # Accept a few variants just in case
    s = re.sub(r"<B(?:OS)?>", "<BOS>", s)   # <B> or <BOS> -> <BOS>
    # If you ever escaped them in HTML:
    s = s.replace("&lt;BOS&gt;", "<BOS>")
    # Drop any stray repeated </s>
    s = s.split("</s>")[0]
    return s

def clean_text(text):
    """
    Cleans the input text by removing double quotes and backslashes.
    
    Args:
        text (str): The input text to be cleaned
        
    Returns:
        str: The cleaned text with all double quotes and backslashes removed
    """
    # Remove backslashes and double quotes
    cleaned_text = text.replace('\\', '').replace('"', '')
    return cleaned_text

# %%
print(clean_text(normalize_markers(gen)))
print(clean_text(jobs[0]['prompt']))

# %%
len(jobs[0]['completion'])

# %%
print("additional_special_tokens:", tok.additional_special_tokens)
print("'<BOS>' in vocab:", "<BOS>" in tok.get_vocab())
print("'<BOS>' pieces:", tok.tokenize("<BOS>"))
print("'<BOS>' id:", tok.convert_tokens_to_ids("<BOS>"))



