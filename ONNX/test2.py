import numpy as np
from rknnlite.api import RKNNLite
from load_datasets import load_dataset, adjust_input_size, build_vocab

# Constants
RKNN_MODEL = './transformer.rknn'
SRC_DATASET = './srcdataset.txt'
REQUIRED_SIZE = 2560  # Adjust this to the model's actual requirement
PADDING_VALUE = 0

def preprocess_input(src_vocab, file_path, required_size, padding_value):
    src_data = load_dataset(file_path)
    src_data = adjust_input_size(src_data, required_size, padding_value)
    return np.array(src_data, dtype=np.float32)

if __name__ == '__main__':
    print('--> Building vocabulary')
    src_vocab = build_vocab(SRC_DATASET)

    print('--> Preprocessing input data')
    input_data = preprocess_input(src_vocab, SRC_DATASET, REQUIRED_SIZE, PADDING_VALUE)

    print('--> Initializing RKNNLite runtime')
    rknn = RKNNLite()

    print('--> Configuring RKNN for debugging')
    rknn.config(verbose=True)  # Enable debug logs

    print('--> Loading RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Failed to load RKNN model')
        exit(ret)
    print('RKNN model loaded successfully.')

    print('--> Initializing runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Failed to initialize RKNN runtime environment')
        exit(ret)
    print('Runtime environment initialized.')

    print('--> Running inference')
    for i, src_input in enumerate(input_data):
        input_tensor = np.expand_dims(src_input, axis=0).astype(np.float32)  # Ensure correct shape and type

        try:
            outputs = rknn.inference(inputs=[input_tensor])
            if outputs is None:
                raise ValueError("Inference returned None. Check input size or model compatibility.")

            print(f"Sample {i + 1}/{len(input_data)}: Output shape: {outputs[0].shape}")
            print(f"Output data: {outputs[0][:5]}")
        except Exception as e:
            print(f"Error during inference for sample {i + 1}: {e}")
            continue

    print('--> Releasing resources')
    rknn.release()
    print('All operations completed successfully.')

