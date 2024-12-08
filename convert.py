from rknn.api import RKNN
import torch
from transformer import Transformer

if __name__ == "__main__":
    DATASET_TXT_PATH = "./tgtdataset.txt"
    MODEL_PATH = './transformer.pt'
    RKNN_MODEL_PATH = './transformer.rknn'

    rknn = RKNN()

    print('--> Configuring RKNN model')
    rknn.config(target_platform='rk3588')
    print('done')

    print('--> Loading PyTorch model')
    model = Transformer(src_vocab_size=3079, tgt_vocab_size=2829)  # Adjust as needed
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    ret = rknn.load_pytorch(model=model, input_size_list=[[32, 20, 512]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=True)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    print(f"RKNN model has been successfully exported to {RKNN_MODEL_PATH}")
