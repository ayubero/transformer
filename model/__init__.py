import torch
import torch.nn as nn

from model.embeddings import InputEmbeddings, PositionalEncoding
from model.blocks import EncoderBlock, Encoder, DecoderBlock, Decoder
from model.layers import ProjectionLayer, FeedForward, MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, encoder_embed: InputEmbeddings, decoder_embed: InputEmbeddings, encoder_pos: PositionalEncoding, decoder_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        self.encoder_pos = encoder_pos
        self.decoder_pos = decoder_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.encoder_embed(src)
        src = self.encoder_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.decoder_embed(tgt)
        tgt = self.decoder_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_seq_len: int, 
        tgt_seq_len: int, 
        d_model: int=512, 
        N: int=6, 
        h: int=8, 
        dropout: float=0.1, 
        d_ff: int=2048
    ) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder & decoder blocks
    encoder_blocks = []
    decoder_blocks = []
    for _ in range(N):
        # Encoder
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

        # Decoder
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer