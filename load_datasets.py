import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import IWSLT2017  

class TranslationDataset(Dataset):
    def __init__(self, data_iter, src_vocab=None, tgt_vocab=None):
        self.data = list(data_iter)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]

        if self.src_vocab:
            src = [self.src_vocab[token] for token in src.split()]
        if self.tgt_vocab:
            tgt = [self.tgt_vocab[token] for token in tgt.split()]

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    src_padded = torch.zeros(len(batch), max(src_lens), dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max(tgt_lens), dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt

    return src_padded, tgt_padded

if __name__ == "__main__":
    train_iter = IWSLT2017(split='train', language_pair=('de', 'en'))
    val_iter = IWSLT2017(split='valid', language_pair=('de', 'en'))

    train_dataset = TranslationDataset(train_iter)
    val_dataset = TranslationDataset(val_iter)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    for src_batch, tgt_batch in train_loader:
        print("Source batch shape:", src_batch.shape)
        print("Target batch shape:", tgt_batch.shape)
        break
