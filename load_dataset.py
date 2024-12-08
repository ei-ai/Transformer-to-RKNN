from datasets import load_dataset
from transformers import BertTokenizer

def generate_dataset_txt(output_path="./dataset.txt", language="", num_samples=1000):
    print("Loading WMT19 dataset...")
    wmt19 = load_dataset("wmt19", "de-en")
    train_data = wmt19["train"]

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    print(f"Generating {num_samples} samples for RKNN...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(train_data):
            if i >= num_samples:
                break
            try:
                text = example["translation"][language]
                tokenized = tokenizer(text, max_length=20, padding='max_length', truncation=True)
                f.write(' '.join(map(str, tokenized["input_ids"])) + "\n")
            except KeyError:
                print(f"KeyError at sample {i}, skipping...")
    print(f"Dataset file saved to {output_path}")

if __name__ == '__main__':
    generate_dataset_txt(output_path="./srcdataset.txt", language="de")
    generate_dataset_txt(output_path="./tgtdataset.txt", language="en")
