import os
import torch
import torchmetrics.text as torchmetrics
from pathlib import Path
from tqdm import tqdm

import config
from dataset import causal_mask
from model import build_transformer
from utils import get_dataset, get_weights_file_path, latest_weights_file_path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # Get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, 'Batch size must be 1 for validation'

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    '''if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation CER', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation WER', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()'''
    
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    print_msg(f"{f'Validation CER:':>15} {cer}")
    print_msg('-'*console_width)

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    print_msg(f"{f'Validation WER:':>15} {wer}")
    print_msg('-'*console_width)

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    print_msg(f"{f'Validation BLEU:':>15} {bleu}")
    print_msg('-'*console_width)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps or torch.backends.mps.is_available() else 'cpu'
    print('Using device:', device)
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f'{config.DATASOURCE}_{config.MODEL_FOLDER}').mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset()
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(),
        config.SEQ_LEN, 
        config.SEQ_LEN, 
        d_model=config.D_MODEL
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, eps=1e-9)
    initial_epoch = 0
    global_step = 0
    preload = config.PRELOAD
    model_filename = latest_weights_file_path() if preload == 'latest' else get_weights_file_path(preload) if preload else None
    batch_iterator = tqdm(val_dataloader, desc=f'Processing validation dataset')
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.SEQ_LEN, device, lambda msg: batch_iterator.write(msg), global_step, None,num_examples=3)
    else:
        print('No model to preload, train a model first')