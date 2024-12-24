from rknn.api import RKNN
import onnx
import torch

def calculate_vocab_size(data_path):
    try:
        data = torch.load(data_path, weights_only=True)
        vocab_size = max(data) + 1  # Assuming vocab indices are 0-indexed
        return vocab_size
    except Exception as e:
        print(f"Error calculating vocabulary size: {e}")
        exit(1)

def get_input_size_from_onnx(onnx_path):
    try:
        model = onnx.load(onnx_path)
        input_tensor = model.graph.input[0]
        input_shape = [
            dim.dim_value if dim.dim_value > 0 else 1  # Replace dynamic dims with 1
            for dim in input_tensor.type.tensor_type.shape.dim
        ]
        return input_shape
    except Exception as e:
        print(f"Error reading ONNX input size: {e}")
        exit(1)

if __name__ == "__main__":
    MODEL_PATH = './transformer.onnx'
    RKNN_MODEL_PATH = './transformer.rknn'
    SRC_DATA_PATH = '../datasets_WMT/src_19_de.pt'  
    TGT_DATA_PATH = '../datasets_WMT/tgt_19_en.pt'  
    
    src_vocab_size = calculate_vocab_size(SRC_DATA_PATH)
    tgt_vocab_size = calculate_vocab_size(TGT_DATA_PATH)

    # Get input size from ONNX model
    input_size = get_input_size_from_onnx(MODEL_PATH)
    input_size_list = [input_size]

    rknn = RKNN()

    print('--> Configuring RKNN model')
    rknn.config(target_platform='rk3588')
    print('done')

    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=MODEL_PATH, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=False)
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
