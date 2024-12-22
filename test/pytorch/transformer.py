import torch
import torch.nn as nn
import math
from collections import Counter

def build_vocab(file_path):
    vocab = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            vocab.update(words)
    vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), start=1)}
    vocab['<PAD>'] = 0  # Ensure <PAD> has unique mapping
    return vocab

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.src_tok_emb(src) * math.sqrt(src.size(-1))
        tgt = self.tgt_tok_emb(tgt) * math.sqrt(tgt.size(-1))
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.fc_out(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

if __name__ == "__main__":
    src_dataset_path = './srcdataset.txt'
    tgt_dataset_path = './tgtdataset.txt'
    src_vocab = build_vocab(src_dataset_path)
    tgt_vocab = build_vocab(tgt_dataset_path)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Source vocab size: {src_vocab_size}, Target vocab size: {tgt_vocab_size}")
    model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    torch.save(model.state_dict(), './transformer.pt')  # Save model
    torch.save(model, './transformer.pth')
    transformer_scripted = torch.jit.script(model) 
    transformer_scripted.save('./transformer_scripted.pt')
    # transformer_traced = torch.jit.trace(model) 
    # transformer_traced.save('./transformer_traced.pt')
