# %%
# !pip install -q --upgrade torch accelerate kernels
# !pip install -q git+https://github.com/huggingface/transformers triton==3.4 git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
# !pip uninstall -q torchvision torchaudio -y

# %%
gpt = True

# %%
import sys
sys.path.append('../src')
import paths

# %%
if gpt:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config

    model_id = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    quantization_config=Mxfp4Config.from_dict(config.quantization_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="cuda",
    )

# %%
if gpt:
    import pandas as pd
    import dataset
    import importlib
    importlib.reload(dataset)
    import gptoss_sent_split
    importlib.reload(gptoss_sent_split)
    from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq

    cfg = BOSConfig(max_new_tokens=256)

    pairs = read_token_label_file(paths.data/"manzoni_train_tokens.csv")
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    y_pred = run_bos_labeling(jobs, model, tokenizer, cfg)

    tokens = [t for (t, _) in pairs]
    gold = [y for (_, y) in pairs]

    # Align lengths, just in case
    n = min(len(tokens), len(y_pred))
    tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    sents = sentences_from_word_seq(tokens, y_pred)
    import pickle
    with open(paths.results/'gptpredtrain.pkl', 'wb') as f:
        pickle.dump(y_pred, f)

# %%
if gpt:
    import pandas as pd
    import dataset
    import importlib
    importlib.reload(dataset)
    import gptoss_sent_split
    importlib.reload(gptoss_sent_split)
    from gptoss_sent_split import BOSConfig, read_token_label_file, build_bos_jobs_by_n_sentences, run_bos_labeling, sentences_from_word_seq

    cfg = BOSConfig(max_new_tokens=256)

    pairs = read_token_label_file(paths.data/"manzoni_dev_tokens.csv")
    jobs = build_bos_jobs_by_n_sentences(pairs, tokenizer, cfg)
    y_pred = run_bos_labeling(jobs, model, tokenizer, cfg)

    tokens = [t for (t, _) in pairs]
    gold = [y for (_, y) in pairs]

    # Align lengths, just in case
    n = min(len(tokens), len(y_pred))
    tokens, gold, y_pred = tokens[:n], gold[:n], y_pred[:n]
    sents = sentences_from_word_seq(tokens, y_pred)
    import pickle
    with open(paths.results/'gptpredval.pkl', 'wb') as f:
        pickle.dump(y_pred, f)


