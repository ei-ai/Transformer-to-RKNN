import torch
from rknn.api import RKNN
import os

MODEL_PATH = './transformer.pt'
RKNN_MODEL_PATH = './transformer.rknn'
DATASET_TXT_PATH = './dataset.txt'

def export_pytorch_model():
    from transformer import Transformer
    model = Transformer()
    model.eval()
    dummy_src = torch.randint(0, 10000, (32, 20))
    dummy_tgt = torch.randint(0, 10000, (32, 20))
    traced_model = torch.jit.trace(model, (dummy_src, dummy_tgt))
    traced_model.save(MODEL_PATH)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("Exporting PyTorch model...")
        export_pytorch_model()

    print("Loading PyTorch model for RKNN conversion...")
    rknn = RKNN()

    print("--> Configuring RKNN")
    rknn.config(mean_values=[0], std_values=[1], target_platform='rk3588')

    print("--> Loading PyTorch model")
    ret = rknn.load_pytorch(model=MODEL_PATH, input_size_list=[[32, 20], [32, 20]])
    if ret != 0:
        print('Failed to load PyTorch model')
        exit(ret)

    print("--> Building RKNN model")
    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT_PATH)
    if ret != 0:
        print('Failed to build RKNN model')
        exit(ret)

    print("--> Exporting RKNN model")
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Failed to export RKNN model')
        exit(ret)

    print("RKNN model has been successfully exported to", RKNN_MODEL_PATH)
    rknn.release()
