from collections import Counter
import os


def build_vocab(file_path):
    vocab = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            vocab.update(words)
    vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), start=1)}
    vocab['<PAD>'] = 0
    return vocab


def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [list(map(int, line.strip().split())) for line in lines]


def adjust_input_size(data, required_size, padding_value=0):
    adjusted_data = []
    for sample in data:
        if len(sample) < required_size:
            sample = sample + [padding_value] * (required_size - len(sample))
        elif len(sample) > required_size:
            sample = sample[:required_size]
        adjusted_data.append(sample)
    return adjusted_data

