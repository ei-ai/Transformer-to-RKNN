import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k, WMT14
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 및 Vocab 구축
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
}

def build_vocab(dataset_iter, tokenizer):
    vocab = build_vocab_from_iterator(map(tokenizer, dataset_iter), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
src_vocab = build_vocab((src for src, tgt in train_iter), token_transform[SRC_LANGUAGE])
tgt_vocab = build_vocab((tgt for src, tgt in train_iter), token_transform[TGT_LANGUAGE])

# 데이터셋 처리 함수
def data_process(raw_text_iter, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
    data = []
    for src, tgt in raw_text_iter:
        src_tensor = torch.tensor([src_vocab[token] for token in src_tokenizer(src)], dtype=torch.long)
        tgt_tensor = torch.tensor([tgt_vocab[token] for token in tgt_tokenizer(tgt)], dtype=torch.long)
        data.append((src_tensor, tgt_tensor))
    return data

train_iter, valid_iter, test_it

