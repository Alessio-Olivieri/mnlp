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

    cfg = BOSConfig(max_new_tokens=10000, n_sentences=3, batch_size=12)

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
