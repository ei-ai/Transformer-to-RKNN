from datasets import load_dataset


def generate_dataset_txt(output_path="./dataset.txt", num_samples=1000):
    print("Loading WMT19 dataset...")
    wmt19 = load_dataset("wmt19", "de-en")

    train_data = wmt19["train"]

    print(f"Generating {num_samples} samples for RKNN...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(train_data):
            if i >= num_samples:
                break
            f.write(example["translation"]["de"] + "\n")

    print(f"Dataset file saved to {output_path}")
    
if __name__ =='__main__':
    generate_dataset_txt()
