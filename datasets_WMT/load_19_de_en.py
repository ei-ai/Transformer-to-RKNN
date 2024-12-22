from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def generate_dataset_txt(output_path="./dataset.pt", language="", num_samples=1000, max_length=50):
    print("Loading WMT19 dataset...")
    wmt19 = load_dataset("wmt19", "de-en")
    train_data = wmt19["train"]

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    print(f"Generating {num_samples} samples for RKNN...")
    all_ids = []
    for i, example in enumerate(train_data):
        if i >= num_samples:
            break
        if "translation" in example and language in example["translation"]:
            text = example["translation"][language]
            tokenized = tokenizer(text, max_length=max_length, padding='max_length', truncation=True)
            all_ids.append(tokenized["input_ids"])
    
    all_ids_tensor = torch.tensor(all_ids)

    print(f"Saving dataset to {output_path}...")
    torch.save(all_ids_tensor, output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == '__main__':
    generate_dataset_txt(output_path="./src_19_de.pt", language="de", num_samples=1000)
    generate_dataset_txt(output_path="./tgt_19_en.pt", language="en", num_samples=1000)

