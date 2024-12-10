import numpy as np
from rknnlite.api import RKNNLite
from load_datasets import load_dataset, adjust_input_size, build_vocab

# Constants
RKNN_MODEL = './transformer.rknn'
SRC_DATASET = './srcdataset.txt'
REQUIRED_SIZE = 2560  # Model's input size (adjust based on model's requirement)
PADDING_VALUE = 0  # Padding value for inputs

def preprocess_input(src_vocab, file_path, required_size, padding_value):
    """
    Preprocess the input data to match the model's requirements.
    :param src_vocab: Vocabulary dictionary for the source dataset.
    :param file_path: Path to the source dataset file.
    :param required_size: Required input size for the model.
    :param padding_value: Padding value to match the required size.
    :return: Processed and adjusted input data.
    """
    src_data = load_dataset(file_path)  # Load raw dataset
    src_data = adjust_input_size(src_data, required_size, padding_value)
    return np.array(src_data, dtype=np.float32)  # Convert to float32 for RKNN model

if __name__ == '__main__':
    print('--> Building vocabulary')
    src_vocab = build_vocab(SRC_DATASET)
    src_vocab_size = len(src_vocab)

    print('--> Preprocessing input data')
    input_data = preprocess_input(src_vocab, SRC_DATASET, REQUIRED_SIZE, PADDING_VALUE)

    print('--> Initializing RKNNLite runtime')
    rknn = RKNNLite()

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
        # Prepare input tensor for inference
        input_tensor = np.expand_dims(src_input, axis=0)  # Add batch dimension

        try:
            # Run inference
            outputs = rknn.inference(inputs=[input_tensor])

            # Print results
            if outputs is None:
                raise ValueError("Inference returned None. Please check model and input compatibility.")
            
            print(f"Sample {i + 1}/{len(input_data)}: Output shape: {outputs[0].shape}")
            print(f"Output data: {outputs[0][:5]}")  # Display first 5 elements of the output
        except Exception as e:
            print(f"Error during inference for sample {i + 1}: {e}")

    print('--> Releasing resources')
    rknn.release()
    print('All operations completed successfully.')

