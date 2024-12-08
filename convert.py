
from rknn.api import RKNN

if __name__ == "__main__":
    DATASET_TXT_PATH = "./dataset.txt"
    MODEL_PATH = './transformer.pt'
    RKNN_MODEL_PATH = './transformer.rknn'

    rknn = RKNN()

    print('--> Configuring RKNN model')
    mean_values = [0] * 20
    std_values = [1] * 20
    rknn.config(mean_values=mean_values, std_values=std_values, target_platform='rk3588')
    print('done')

    print('--> Loading PyTorch model')
    ret = rknn.load_pytorch(model=MODEL_PATH, input_size_list=[[32, 20]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT_PATH)
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
