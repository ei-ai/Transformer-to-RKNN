import numpy as np
import torch
from rknn.api import RKNN
from transformer import TransformerModel
from load_datasets import src_vocab, tgt_vocab, data_process, token_transform
import os

def export_transformer_model():
    # Transformer 모델 초기화
    input_dim = len(src_vocab)
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6
    dropout = 0.1
    
    model = TransformerModel(input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout)
    model.eval()
    
    # 임의의 입력 데이터로 TorchScript 변환
    dummy_input = torch.randint(0, input_dim, (1, 50))  # (batch_size, sequence_length)
    trace_model = torch.jit.trace(model, dummy_input)
    trace_model.save('./transformer.pt')


def show_outputs(output, vocab):
    index = torch.argmax(output, dim=-1).squeeze(0).tolist()
    output_str = "Decoded Sequence:\n" + " ".join(vocab.itos[idx] for idx in index)
    print(output_str)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


if __name__ == '__main__':
    # Transformer 모델 파일 경로
    model_path = './transformer.pt'
    if not os.path.exists(model_path):
        export_transformer_model()
    
    input_size_list = [[1, 50]]  # Transformer 입력 크기 (배치, 시퀀스 길이)
    rknn = RKNN(verbose=True)
    
    # RKNN 설정
    print('--> Config model')
    rknn.config(target_platform='rk3588')  # 필요시 mean_values와 std_values 추가 가능
    print('done')

    # PyTorch 모델 로드
    print('--> Loading Transformer model')
    ret = rknn.load_pytorch(model=model_path, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # RKNN 모델 빌드
    print('--> Building model')
    ret = rknn.build(do_quantization=False)  # 양자화 없이 진행
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # RKNN 모델 내보내기
    print('--> Export rknn model')
    ret = rknn.export_rknn('./transformer.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 데이터 로드 및 전처리
    print('--> Preparing input data')
    sample_sentence = "Hallo Welt!"  # 예제 문장
    src_tokenizer = token_transform['de']
    tgt_vocab = tgt_vocab
    src_tensor = torch.tensor([src_vocab[token] for token in src_tokenizer(sample_sentence)], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
    print('done')

    # RKNN 런타임 환경 초기화
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 추론 수행
    print('--> Running Transformer model')
    outputs = rknn.inference(inputs=[src_tensor.numpy()])
    np.save('./transformer_output.npy', outputs[0])
    show_outputs(torch.tensor(outputs[0]), tgt_vocab)
    print('done')

    # RKNN 객체 해제
    rknn.release()
