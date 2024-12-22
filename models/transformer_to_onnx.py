import torch
import torch.nn as nn
from transformer import Transformer

src_PATH = '../datasets_WMT/src_19_de.pt' 
tgt_PATH = '../datasets_WMT/tgt_19_en.pt' 

def export_to_onnx(model, src_vocab_size, tgt_vocab_size, file_name="transformer.onnx"):
    dummy_src = torch.randint(0, src_vocab_size, (1, 10))
    dummy_tgt = torch.randint(0, tgt_vocab_size, (1, 10))
    torch.onnx.export(model, (dummy_src, dummy_tgt), file_name, opset_version=13, input_names=['src', 'tgt'], output_names=['output'])

def main():
    src_vocab_size = 32000
    tgt_vocab_size = 32000
    model = Transformer(src_vocab_size, tgt_vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    print("Load datasets...")
    src_data = torch.load(src_PATH)  # Tokenized source dataset
    tgt_data = torch.load(tgt_PATH)  # Tokenized target dataset
    print(f"src_data min: {src_data.min()}, max: {src_data.max()}")
    print(f"tgt_data min: {tgt_data.min()}, max: {tgt_data.max()}")
    src_data = torch.clamp(src_data, 0, src_vocab_size - 1)
    tgt_data = torch.clamp(tgt_data, 0, tgt_vocab_size - 1)
    print("done")

    print("Training model...")
    for epoch in range(10):  # Example training loop
        model.train()
        optimizer.zero_grad()

        output = model(src_data, tgt_data[:, :-1])  # Shifted target for teacher forcing
        loss = criterion(output.view(-1, tgt_vocab_size), tgt_data[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    print("done")

    print("Export to ONNX...")
    export_to_onnx(model, src_vocab_size, tgt_vocab_size)
    print("done")

if __name__ == "__main__":
    main()
