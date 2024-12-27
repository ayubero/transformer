import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask
import config

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path(config.TOKENIZER_FILE.format(lang))
    # Code taken from https://huggingface.co/docs/tokenizers/quicktour
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset():
    # It only has the train split, so we divide it overselves
    dataset_raw = load_dataset(f"{config.DATASOURCE}", f"{config.LANG_SRC}-{config.LANG_TGT}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(dataset_raw, config.LANG_SRC)
    tokenizer_tgt = get_or_build_tokenizer(dataset_raw, config.LANG_TGT)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config.LANG_SRC, config.LANG_TGT, config.SEQ_LEN)
    val_ds = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config.LANG_SRC, config.LANG_TGT, config.SEQ_LEN)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config.LANG_SRC]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config.LANG_TGT]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt