import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset
import config

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(dataset, lang):
    tokenizer_path = Path(config.TOKENIZER_FILE.format(lang))
    # Code taken from https://huggingface.co/docs/tokenizers/quicktour
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset():
    # It only has the train split, so we divide it overselves
    dataset_raw = load_dataset(f'{config.DATASOURCE}', f'{config.LANG_SRC}-{config.LANG_TGT}', split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(dataset_raw, config.LANG_SRC)
    tokenizer_tgt = get_or_build_tokenizer(dataset_raw, config.LANG_TGT)

    # Filter out sentences longer than SEQ_LEN tokens
    def filter_long_sentences(item):
        src_len = len(tokenizer_src.encode(item['translation'][config.LANG_SRC]).ids)
        tgt_len = len(tokenizer_tgt.encode(item['translation'][config.LANG_TGT]).ids)
        return src_len <= config.SEQ_LEN and tgt_len <= config.SEQ_LEN

    dataset_filtered = dataset_raw.filter(filter_long_sentences)

    # Keep 90% for training, 10% for validation
    train_dataset_size = int(0.9 * len(dataset_filtered))
    val_dataset_size = len(dataset_filtered) - train_dataset_size
    train_dataset, val_dataset = random_split(dataset_filtered, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(
        train_dataset, tokenizer_src, tokenizer_tgt, config.LANG_SRC, config.LANG_TGT, config.SEQ_LEN
    )
    val_dataset = BilingualDataset(
        val_dataset, tokenizer_src, tokenizer_tgt, config.LANG_SRC, config.LANG_TGT, config.SEQ_LEN
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_filtered:
        src_ids = tokenizer_src.encode(item['translation'][config.LANG_SRC]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config.LANG_TGT]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_weights_file_path(epoch: str):
    model_folder = f'{config.DATASOURCE}_{config.MODEL_FOLDER}'
    model_filename = f'{config.MODEL_PATH}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path():
    model_folder = f'{config.DATASOURCE}_{config.MODEL_FOLDER}'
    model_filename = f'{config.MODEL_PATH}*'
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])