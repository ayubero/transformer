import torch
import torch.nn as nn
#import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import warnings
from pathlib import Path
from tqdm import tqdm

import config
from model import build_transformer
from utils import get_dataset, get_weights_file_path, latest_weights_file_path
from validation import run_validation

def train_model():
    # Define the device
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

    # Tensorboard
    writer = SummaryWriter(config.EXPERIMENT_NAME)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config.PRELOAD
    model_filename = latest_weights_file_path() if preload == 'latest' else get_weights_file_path(preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, config.NUM_EPOCHS):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (batch, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.SEQ_LEN, device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train_model()