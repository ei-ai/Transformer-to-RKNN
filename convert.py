from rknn.api import RKNN
import torch
from transformer import Transformer

if __name__ == "__main__":
    DATASET_TXT_PATH = "./tgtdataset.txt"
    MODEL_PATH = './transformer_traced.pt' 
    RKNN_MODEL_PATH = './transformer.rknn'

    src_vocab_size = 3079  
    tgt_vocab_size = 2829  

    rknn = RKNN()

    scripted_model = torch.jit.load('./transformer_scripted.pt')
    print(scripted_model)

    traced_model = torch.jit.load('./transformer_traced.pt')
    print(traced_model)

    print('--> Configuring RKNN model')
    rknn.config(target_platform='rk3588')
    print('done')

    
    print('--> Loading PyTorch model')
    model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    model.load_state_dict(torch.load('./transformer.pt')) 
    model.eval()

    ret = rknn.load_pytorch(model=MODEL_PATH, input_size_list=[[32, 20, 512]])
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
