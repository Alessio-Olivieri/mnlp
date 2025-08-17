# gptoss_sent_split.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import json, math, re
from dataclasses import dataclass
from pathlib import Path


from dataset import read_token_label_file, group_into_sentences, sentences_from_word_seq

def detok_with_offsets(tokens: List[str]) -> Tuple[str, List[int]]:
    """
    Same logic as your detok(), but also returns char start offsets for each token
    in the produced string. Offsets allow us to align <BOS> insertions back to tokens.
    """
    no_space_before = set(list(".,;:!?)]}%") + ["”", "»", "…", "”", "’", "”"])
    no_space_after_open = set(list("([{") + ["“", "«"])

    out = ""
    starts: List[int] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            starts.append(len(out))
            out += tok
            continue

        prev_char = out[-1] if out else ""
        if tok in no_space_before:
            starts.append(len(out))
            out += tok
        elif tok in {"'", "’"}:
            starts.append(len(out))
            out += tok
        elif prev_char in no_space_after_open:
            starts.append(len(out))
            out += tok
        else:
            starts.append(len(out) + 1)
            out += " " + tok

    return out, starts

SPECIAL_MARKER = "<BOS>"

SYSTEM_PROMPT = (
    f"Rewrite the given text, inserting the token {SPECIAL_MARKER} before each sentence.\n"
    "Rules:\n"
    " - Keep ALL characters from the input unchanged.\n"
    " - Do not add or remove any characters other than inserting the marker.\n"
    f" - Insert {SPECIAL_MARKER} immediately before the FIRST non-space character of each sentence.\n"
    " - A sentence ends with ., !, or ? (possibly followed by quotes or brackets).\n"
    "Output only the rewritten text."
)

def _one_shot_messages(text: str) -> List[Dict[str, str]]:
    demo_in = "Andando , guardava innanzi , ansioso insieme e timoroso di veder qualcheduno; e , dopo pochi passi , vide infatti un uomo in camicia , seduto in terra , con le spalle appoggiate a una siepe di gelsomini , in un' attitudine d' insensato: e , a questa , e poi anche alla fisonomia , gli parve di raffigurar quel povero mezzo scemo di Gervaso ch' era venuto per secondo testimonio alla sciagurata spedizione. Ma essendosegli avvicinato , dovette accertarsi ch' era in vece quel Tonio così sveglio che ce l' aveva condotto. La peste , togliendogli il vigore del corpo insieme e della mente , gli aveva svolto in faccia e in ogni suo atto un piccolo e velato germe di somiglianza che aveva con l' incantato fratello. «Oh Tonio!» gli disse Renzo , fermandosegli davanti: «sei tu?» Tonio alzò gli occhi , senza mover la testa."
    demo_out = f"<BOS>Andando , guardava innanzi , ansioso insieme e timoroso di veder qualcheduno; e , dopo pochi passi , vide infatti un uomo in camicia , seduto in terra , con le spalle appoggiate a una siepe di gelsomini , in un' attitudine d' insensato: e , a questa , e poi anche alla fisonomia , gli parve di raffigurar quel povero mezzo scemo di Gervaso ch' era venuto per secondo testimonio alla sciagurata spedizione. <BOS>Ma essendosegli avvicinato , dovette accertarsi ch' era in vece quel Tonio così sveglio che ce l' aveva condotto. <BOS>La peste , togliendogli il vigore del corpo insieme e della mente , gli aveva svolto in faccia e in ogni suo atto un piccolo e velato germe di somiglianza che aveva con l' incantato fratello. <BOS> «Oh Tonio!» gli disse Renzo , fermandosegli davanti: «sei tu?» <BOS>Tonio alzò gli occhi , senza mover la testa."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": demo_in},
        {"role": "assistant", "content": demo_out},
        {"role": "user", "content": text},
    ]


@dataclass
class BOSConfig:
    model_id: str = "openai/gpt-oss-20b"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    n_sentences: int = 5           # sentences per sample (chunk)
    overlap_sentences: int = 0     # optional context overlap
    token_budget: int = 60000      # prompt-side token cap (for safety)
    safety_margin: int = 2048      # leave room
    retry_on_mismatch: bool = True
    batch_size: int = 8
    

# =========================
# Chunking by N sentences
# =========================

def _prompt_len(tokenizer, messages: List[Dict[str, str]]) -> int:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return len(ids)

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

def build_bos_jobs_by_n_sentences(
    pairs: List[Tuple[str, int]],
    tokenizer,
    cfg: BOSConfig,
) -> List[Dict[str, Any]]:
    """
    Creates jobs so each contains ~cfg.n_sentences sentences (with overlap), while
    respecting a prompt token budget.
    Each job has: {messages, start, tokens, text, starts}
      - tokens: flat list of tokens in the chunk
      - text: detokenized text for the chunk
      - starts: char offsets for tokens in 'text' (from detok_with_offsets)
      - start: global token index offset of first token in this chunk
    """
    s_tokens, _ = group_into_sentences(pairs)
    words = [t for (t, _) in pairs]

    # Prefix sums to map sentence windows to flat indices
    prefix = [0]
    for s in s_tokens:
        prefix.append(prefix[-1] + len(s))

    S = len(s_tokens)
    jobs: List[Dict[str, Any]] = []

    i = 0
    step = max(1, cfg.n_sentences - cfg.overlap_sentences)
    max_prompt = max(512, cfg.token_budget - cfg.safety_margin - cfg.max_new_tokens)

    while i < S:
        k = min(cfg.n_sentences, S - i)
        # shrink window until prompt fits
        while k > 0:
            start_w = prefix[i]
            end_w = prefix[i + k]
            chunk_tokens = words[start_w:end_w]
            text, starts = detok_with_offsets(chunk_tokens)
            msgs = _one_shot_messages(clean_text(text))
            if _prompt_len(tokenizer, msgs) <= max_prompt:
                jobs.append({
                    "messages": msgs,
                    "start": start_w,
                    "tokens": chunk_tokens,
                    "text": clean_text(text),
                    "starts": starts,
                })
                break
            k -= 1

        if k == 0:
            # Fallback: extremely long sentence — slice by words to fit
            # (rare; still keeps BOS mapping consistent within this sub-slice)
            start_w = prefix[i]
            end_w = prefix[i + 1]
            sub = words[start_w:end_w]
            lo = 0
            while lo < len(sub):
                hi = lo + 1
                last_good_hi = lo + 1
                while hi <= len(sub):
                    t, _s = detok_with_offsets(sub[lo:hi])
                    msgs = _one_shot_messages(clean_text(t))
                    if _prompt_len(tokenizer, msgs) <= max_prompt:
                        last_good_hi = hi
                        hi += 1
                    else:
                        break
                t, s_off = detok_with_offsets(sub[lo:last_good_hi])
                jobs.append({
                    "messages": _one_shot_messages(clean_text(t)),
                    "start": start_w + lo,
                    "tokens": sub[lo:last_good_hi],
                    "text": clean_text(t),
                    "starts": s_off,
                })
                lo = last_good_hi
            i += 1
        else:
            i += step

    return jobs


# =========================
# Generation + BOS -> labels
# =========================

def _clean_equal(a: str, b: str) -> bool:
    """Looser equality that collapses whitespace."""
    norm = lambda s: " ".join(s.split())
    return norm(a) == norm(b)

from rapidfuzz import fuzz

    # Flag as mismatch
def _map_bos_markers_to_sentence_starts(marked_text: str, orig_text: str, starts: List[int], model_id) -> List[int]:
    """
    Given marked_text (with <BOS> inserted), original text, and token char start offsets,
    return a list of token indices that begin sentences (0 included if first sentence marked).
    """
    # Remove markers and verify we still match original (strict or loose)
    cleaned = marked_text.replace(SPECIAL_MARKER, "")
    # print("\n\n_map_bos_markers_to_sentence_starts\n\n", cleaned, "\n\n")
    if model_id == "openai/gpt-oss-20b":
        cleaned = extract_gpt_answer(cleaned)
        
    # print("\n\n_map_bos_markers_to_sentence_starts\n\n", cleaned, "\n\n")
    if fuzz.token_sort_ratio(orig_text, cleaned) < 80 or len(orig_text.split()) != len(cleaned.split()):
        return []  # caller will handle fallback

    # Find BOS positions in marked_text, map to original text positions
    bos_positions = [m.start() for m in re.finditer(re.escape(SPECIAL_MARKER), marked_text)]
    start_token_ids: List[int] = []

    for i, pos in enumerate(bos_positions):
        # subtract previously inserted marker lengths to map into orig_text coords
        corrected = pos - i * len(SPECIAL_MARKER)
        # find the token whose start >= corrected
        # (markers are inserted right before the first char of the sentence)
        idx = None
        for ti, st in enumerate(starts):
            if st == corrected:
                idx = ti
                break
            if st > corrected:
                idx = ti  # just in case spacing differences push by 1
                break
        if idx is None:
            # if not found, attach to nearest start on the right
            idx = len(starts) - 1
        start_token_ids.append(idx)

    # Ensure uniqueness and sort
    start_token_ids = sorted(set(start_token_ids))
    return start_token_ids


def _labels_from_sentence_starts(n_tokens: int, sent_starts: List[int]) -> List[int]:
    labels = [0] * n_tokens
    if not sent_starts:
        labels[-1] = 1
        return labels
    # The first start is start of text (ideally 0). We set boundaries before subsequent starts.
    for s in sent_starts:
        if s > 0:
            labels[s - 1] = 1
    # Always close the last sentence in the chunk
    labels[-1] = 1
    return labels

def _fallback_punct_labels_2(tokens: List[str]) -> List[int]:
    # Primary sentence-ending punctuations
    enders = {".", "!", "?", "…"}
    # Secondary punctuations that sometimes end sentences
    secondary_enders = {";", ":"}
    # Common Italian abbreviations that contain periods
    abbreviations = {"sig", "sig.ra", "dott", "prof", "ing", "arch", "avv", "etc", 
                     "es", "p.es", "ecc", "art", "n", "pp", "ss", "ca", "c.a", 
                     "s.a", "s.p.a", "s.r.l", "d", "s", "c", "e", "l", "m", "p", "t"}
    
    labels = [0] * len(tokens)
    
    for i, t in enumerate(tokens):
        # Check if token is a primary ending punctuation
        if t in enders:
            # Special handling for periods to avoid abbreviations
            if t == ".":
                # Check if this is part of an abbreviation
                if i > 0 and tokens[i-1].lower() in abbreviations:
                    continue  # Skip if it's part of an abbreviation
                
                # Check if next token is lowercase (likely continuation of sentence)
                if i+1 < len(tokens) and tokens[i+1] and tokens[i+1][0].islower():
                    continue  # Skip if next word starts with lowercase
            
            labels[i] = 1
        
        # Check if token is a secondary ending punctuation
        elif t in secondary_enders:
            # Only mark as boundary if followed by uppercase or end of sequence
            if i+1 >= len(tokens) or (tokens[i+1] and tokens[i+1][0].isupper()):
                labels[i] = 1
    
    # Always mark the last token as boundary
    if labels and labels[-1] == 0:
        labels[-1] = 1
    
    return labels

def _fallback_punct_labels(tokens: List[str]) -> List[int]:
    enders = {".", "!", "?", "…"}
    labels = [0] * len(tokens)
    for i, t in enumerate(tokens):
        if t in enders:
            labels[i] = 1
    if labels and labels[-1] == 0:
        labels[-1] = 1
    return labels

from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import warnings
import torch

from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import warnings
import torch

def extract_gpt_answer(output):
    marker = "assistantfinal"
    if marker in output:
        after = output.split(marker, 1)[1]
        return after
    else:
        print("Marker not found.")

def run_bos_labeling(
    jobs: List[Dict[str, Any]],
    model,
    tokenizer,
    cfg,
    model_id=None
) -> Tuple[List[int], List[int]]:
    """
    Batched BOS labeling with logging for skipped-after-retry sentences.

    Returns:
        y_pred: List[int] - predicted labels for the full sequence
        skipped_jobs: List[int] - indices of jobs that failed even after retry
    """
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        if model.generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.eos_token_id

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = getattr(cfg, "batch_size", 8)

    prompts, nudged_prompts = [], []
    for job in jobs:
        base_p = tokenizer.apply_chat_template(job["messages"], tokenize=False, add_generation_prompt=True)
        prompts.append(base_p)
        if getattr(cfg, "retry_on_mismatch", False):
            nudged_msgs = list(job["messages"])
            nudged_msgs[-1]["content"] += (
                f"\nReminder: Insert {SPECIAL_MARKER} before EACH sentence. "
                "Do not change any other characters or words. Output only the rewritten text."
            )
            nudged_prompts.append(
                tokenizer.apply_chat_template(nudged_msgs, tokenize=False, add_generation_prompt=True)
            )
        else:
            nudged_prompts.append(base_p)

    def _generate_batched(ids: List[int], retry: bool) -> List[str]:
        sel = [nudged_prompts[i] if retry else prompts[i] for i in ids]
        inputs = tokenizer(sel, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(gen, skip_special_tokens=True)

    preds_full, skipped_jobs, needs_retry = {}, [], []
    skipped_texts_after_retry: List[str] = {}

    # First pass
    for st in tqdm(range(0, len(jobs), batch_size), desc="BOS labeling (pass 1)", unit="batch"):
        batch = list(range(st, min(st + batch_size, len(jobs))))
        outs = _generate_batched(batch, retry=False)
        for idx_local, job_idx in enumerate(batch):
            job = jobs[job_idx]
            marked = outs[idx_local]
            starts = _map_bos_markers_to_sentence_starts(marked, job["text"], job["starts"], model_id)
            if not starts and getattr(cfg, "retry_on_mismatch", False):
                needs_retry.append(job_idx)
            elif not starts:
                skipped_jobs.append(job_idx)
                warnings.warn(f"Skipped BOS labeling for job {job_idx}")
            else:
                labels = _labels_from_sentence_starts(len(job["tokens"]), starts)
                for i, y in enumerate(labels):
                    preds_full[job["start"] + i] = y

    # Retry pass
    if needs_retry:
        for st in tqdm(range(0, len(needs_retry), batch_size), desc="BOS labeling (retry)", unit="batch"):
            batch = needs_retry[st: st + batch_size]
            outs = _generate_batched(batch, retry=True)
            for idx_local, job_idx in enumerate(batch):
                job = jobs[job_idx]
                marked = outs[idx_local]
                starts = _map_bos_markers_to_sentence_starts(marked, job["text"], job["starts"], model_id)
                if not starts:
                    skipped_jobs.append(job_idx)
                    skipped_texts_after_retry[job_idx] = marked
                    warnings.warn(f"Skipped BOS labeling for job {job_idx} (after retry)")
                else:
                    labels = _labels_from_sentence_starts(len(job["tokens"]), starts)
                    for i, y in enumerate(labels):
                        preds_full[job["start"] + i] = y

    # Log the skipped-after-retry cases
    if skipped_texts_after_retry:
        print("\n--- Skipped-after-retry sentences detail ---")
        for job_idx, text in skipped_texts_after_retry.items():
            print(f"[Job GOLD{job_idx}]:\n{jobs[job_idx]["text"]}\n")
            print(f"[Job PREDICTION{job_idx}]:\n{text}\n")

    max_idx = max(preds_full.keys()) if preds_full else -1
    y_pred = [preds_full.get(i, 0) for i in range(max_idx + 1)]
    return y_pred, skipped_jobs




# =========================
# Public API
# =========================

def predict_boundaries_with_bos(
    csv_path: str | Path,
    model,
    tokenizer,
    cfg: BOSConfig = BOSConfig(),
) -> Dict[str, Any]:
    pairs = read_token_label_file(csv_path)
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    y_pred = run_bos_labeling(jobs, model, tokenizer, cfg)

    tokens = [t for (t, _) in pairs]
    gold = [y for (_, y) in pairs]

    # Align lengths, just in case
    n = min(len(tokens), len(y_pred))
    tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    sents = sentences_from_word_seq(tokens, y_pred)

    return {
        "tokens": tokens,
        "gold": gold,
        "pred": y_pred,
        "sentences": sents,
    }