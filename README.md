# Transformer-to-RKNN

## 레포지토리 기능
1. **트랜스포머 모델의 구현**: rknn_toolkit2 라이브러리에서 지원하는 연산자와 파이썬 라이브러리로 구현해야 한다.
2. **rknn 모델 변환**: rknn_toolkit2 라이브러리를 이용해, 구현한 트랜스포머 모델을 `.rknn` 모델로 변환한다.
3. **rknnpu에서 모델 실행**: rknn_toolkit_lite2 라이브러리를 이용해 rknn 모델을 npu 보드에서 실행한다.

## 레포지토리 구조
<details>
<summary>Structure</summary>
<div markdown=1>

📦Transformer-to-RKNN  
 ┣ 📂datasets_WMT
 ┃ ┣ 📜load_19_de_en.py
 ┃ ┣ 📜src_19_de.pt
 ┃ ┗ 📜tgt_19_en.pt
 ┣ 📂models
 ┃ ┣ 📜check.py
 ┃ ┣ 📜onnx_to_rknn.py
 ┃ ┣ 📜transformer.onnx
 ┃ ┣ 📜transformer.py
 ┃ ┣ 📜transformer.rknn
 ┃ ┗ 📜transformer_to_onnx.py
 ┣ 📂test
 ┃ ┣ 📂ONNX
 ┃ ┃ ┣ 📜convertONNX.py
 ┃ ┃ ┣ 📜test.py
 ┃ ┃ ┣ 📜transformer.rknn
 ┃ ┃ ┗ 📜transformerONNX.py
 ┃ ┣ 📂pytorch
 ┃ ┃ ┣ 📜convert.py
 ┃ ┃ ┗ 📜transformer.py
 ┃ ┣ 📜convert.py
 ┃ ┣ 📜load_datasets.py
 ┃ ┣ 📜test.py
 ┃ ┗ 📜transformer.py
 ┗ 📜README.md

</div>
</details>

## 코드 상세
1. [datasets](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/datasets_WMT)
    * `load_19_de_en.py`: 허깅페이스의 데이터셋을 불러오고, BertTokenizer를 사용해 데이터를 토크나이징해서 별개의 파일로 저장한다.
    * 현재는 WMT19 en-de 데이터셋을 샘플 데이터셋으로 사용하고 있다. (src:de, tgt:en)
    * 추후 데이터셋의 추가와 이에 따른 파일명 수정이 필요해 보인다.
    
2. [ONNX](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/ONNX)
    * `transformer.py`: 트랜스포머 모델 구현 및 저장. 
      * 데이터셋은 위에서 저장한 텍스트 파일을 이용한다.
      * 트랜스포머 모델 구현은 파이토치를 사용하되, `aten::unflatten`이나 `aten::scaled_dot_product_attention`과 같이 rknn_toolkit2에서 지원하지 않는 연산은 수학적으로 구현한다.
    * `convert_to_onnx.py`, `convert_to_rknn.py`: PyTorch > ONNX > RKNN모델로 변환.
    * `run.py`: npu 보드에서 rknn 모델을 실행하고 성능을 측정하는 코드 

3. [test](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/test)
    * 테스트/기록용 더미 파일이다.
