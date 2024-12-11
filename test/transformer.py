import torch
import torch.nn as nn
import math
from torch.optim import Adam
from load_datasets import build_vocab, load_dataset, adjust_input_size


class SimplifiedAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SimplifiedAttention, self).__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model / self.nhead)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return self.out(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SimplifiedAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.src_tok_emb(src) * math.sqrt(src.size(-1))
        src = self.positional_encoding(src)

        for layer in self.encoder_layers:
            src = layer(src)

        return self.fc_out(src)


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
    src_vocab = build_vocab('./srcdataset.txt')
    tgt_vocab = build_vocab('./tgtdataset.txt')
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = Transformer(src_vocab_size, tgt_vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<PAD>'])
    optimizer = Adam(model.parameters(), lr=0.001)

    src_data = load_dataset('./srcdataset.txt')
    tgt_data = load_dataset('./tgtdataset.txt')

    src_data = adjust_input_size(src_data, required_size=20, padding_value=src_vocab['<PAD>'])
    tgt_data = adjust_input_size(tgt_data, required_size=20, padding_value=tgt_vocab['<PAD>'])

    src_tensor = torch.tensor(src_data)
    tgt_tensor = torch.tensor(tgt_data)

    for epoch in range(10):
        for i in range(0, len(src_tensor), 32):
            src_batch = src_tensor[i:i + 32]
            tgt_batch = tgt_tensor[i:i + 32]

            optimizer.zero_grad()
            output = model(src_batch, tgt_batch[:, :-1])
            loss = criterion(output.view(-1, tgt_vocab_size), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Batch {i // 32 + 1}, Loss: {loss.item()}")

    torch.onnx.export(
        model,
        (src_tensor[:1], tgt_tensor[:1]),
        "transformer.onnx",
        input_names=["src", "tgt"],
        output_names=["output"],
        opset_version=13
    )
    print("Transformer model saved to transformer.onnx")

