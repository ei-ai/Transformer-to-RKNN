from datasets import load_dataset
from transformers import BertTokenizer

def generate_dataset_txt(output_path="./dataset.txt", num_samples=1000):
    print("Loading WMT19 dataset...")
    wmt19 = load_dataset("wmt19", "de-en")
    train_data = wmt19["train"]

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    print(f"Generating {num_samples} samples for RKNN...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(train_data):
            if i >= num_samples:
                break
            # 독일어 문장 토크나이즈
            tokenized = tokenizer.encode(example["translation"]["de"], truncation=True, padding="max_length", max_length=20)
            f.write(' '.join(map(str, tokenized)) + "\n")

    print(f"Dataset file saved to {output_path}")
    
if __name__ =='__main__':
    generate_dataset_txt()
