# gptoss_sent_split.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import json, math
from dataclasses import dataclass

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
    demo_in = "Va bene? Sì, grazie."
    demo_out = f"{SPECIAL_MARKER}Va bene? {SPECIAL_MARKER}Sì, grazie."
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

# =========================
# Chunking by N sentences
# =========================

def _prompt_len(tokenizer, messages: List[Dict[str, str]]) -> int:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return len(ids)

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
            msgs = _one_shot_messages(text)
            if _prompt_len(tokenizer, msgs) <= max_prompt:
                jobs.append({
                    "messages": msgs,
                    "start": start_w,
                    "tokens": chunk_tokens,
                    "text": text,
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
                    msgs = _one_shot_messages(t)
                    if _prompt_len(tokenizer, msgs) <= max_prompt:
                        last_good_hi = hi
                        hi += 1
                    else:
                        break
                t, s_off = detok_with_offsets(sub[lo:last_good_hi])
                jobs.append({
                    "messages": _one_shot_messages(t),
                    "start": start_w + lo,
                    "tokens": sub[lo:last_good_hi],
                    "text": t,
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

def _map_bos_markers_to_sentence_starts(marked_text: str, orig_text: str, starts: List[int]) -> List[int]:
    """
    Given marked_text (with <BOS> inserted), original text, and token char start offsets,
    return a list of token indices that begin sentences (0 included if first sentence marked).
    """
    # Remove markers and verify we still match original (strict or loose)
    cleaned = marked_text.replace(SPECIAL_MARKER, "")
    if cleaned != orig_text and not _clean_equal(cleaned, orig_text):
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


def _fallback_punct_labels(tokens: List[str]) -> List[int]:
    enders = {".", "!", "?", "…"}
    labels = [0] * len(tokens)
    for i, t in enumerate(tokens):
        if t in enders:
            labels[i] = 1
    if labels and labels[-1] == 0:
        labels[-1] = 1
    return labels


def run_bos_labeling(
    jobs: List[Dict[str, Any]],
    model,
    tokenizer,
    cfg: BOSConfig,
) -> List[int]:
    """
    Runs the BOS-rewrite prompt per job and stitches labels for the full sequence.
    Later chunks overwrite earlier ones on overlaps (safe since boundaries match).
    """
    # generation config niceties
    if getattr(model, "generation_config", None) is not None:
        if model.generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.eos_token_id

    preds_full: Dict[int, int] = {}

    for job in jobs:
        prompt = tokenizer.apply_chat_template(job["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            top_p=cfg.top_p,
        )
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        marked_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Map BOS markers to sentence starts
        sent_starts = _map_bos_markers_to_sentence_starts(marked_text, job["text"], job["starts"])

        if not sent_starts and cfg.retry_on_mismatch:
            # Gentle nudge retry: append a clarifying user turn
            nudged = list(job["messages"]) + [{
                "role": "user",
                "content": (
                    f"Reminder: Insert {SPECIAL_MARKER} before EACH sentence. "
                    "Do not change any other characters. Output only the rewritten text."
                )
            }]
            prompt2 = tokenizer.apply_chat_template(nudged, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
            out_ids2 = model.generate(
                **inputs2,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=(cfg.temperature > 0),
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            gen_ids2 = out_ids2[:, inputs2["input_ids"].shape[1]:]
            marked_text2 = tokenizer.decode(gen_ids2[0], skip_special_tokens=True)
            sent_starts = _map_bos_markers_to_sentence_starts(marked_text2, job["text"], job["starts"])

        if not sent_starts:
            labels = _fallback_punct_labels(job["tokens"])
        else:
            labels = _labels_from_sentence_starts(len(job["tokens"]), sent_starts)

        # Fill into global index space
        for i, y in enumerate(labels):
            preds_full[job["start"] + i] = y

    # Stitch back in order; default to 0 if any holes (shouldn't happen)
    max_idx = max(preds_full.keys()) if preds_full else -1
    y_pred = [preds_full.get(i, 0) for i in range(max_idx + 1)]
    return y_pred


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