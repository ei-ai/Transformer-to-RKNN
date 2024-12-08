from datasets import load_dataset

# RKNN용 데이터셋 파일 생성
def generate_dataset_txt(output_path="./dataset.txt", num_samples=1000):
    print("Loading WMT19 dataset...")
    wmt19 = load_dataset("wmt19", "de-en")

    # 'train' split의 독일어 데이터를 사용
    train_data = wmt19["train"]

    print(f"Generating {num_samples} samples for RKNN...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(train_data):
            if i >= num_samples:
                break
            # 독일어 텍스트 입력만 저장
            f.write(example["translation"]["de"] + "\n")

    print(f"Dataset file saved to {output_path}")

if __name__ == "__main__":
    # 데이터셋 파일 생성
    DATASET_TXT_PATH = "./dataset.txt"
    generate_dataset_txt(output_path=DATASET_TXT_PATH)

    # RKNN 변환 코드 실행 (기존 코드 유지)
    from rknn.api import RKNN

    MODEL_PATH = './transformer.pt'
    RKNN_MODEL_PATH = './transformer.rknn'

    # Create RKNN object
    rknn = RKNN()

    # Pre-process config
    print('--> Configuring RKNN model')
    rknn.config(mean_values=[0], std_values=[1], target_platform='rk3588')
    print('done')

    # Load model
    print('--> Loading PyTorch model')
    ret = rknn.load_pytorch(model=MODEL_PATH, input_size_list=[[32, 20]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    print(f"RKNN model has been successfully exported to {RKNN_MODEL_PATH}")
