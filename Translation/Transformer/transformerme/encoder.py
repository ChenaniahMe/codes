import torch.nn as nn

from config import pad_id, n_src_vocab
from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from .utils import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, n_src_vocab=n_src_vocab, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=512, d_inner=2048, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        # parameters
        self.n_src_vocab = n_src_vocab
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        self.src_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x T
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []

        # Forward
        enc_outputs = self.src_emb(padded_input)
        enc_outputs += self.pos_emb(enc_outputs)
        enc_output = self.dropout(enc_outputs)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(enc_output, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(enc_output, input_lengths, length)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
