# Transformer-to-RKNN

## 레포지토리 기능
1. **트랜스포머 모델의 구현**: rknn_toolkit2 라이브러리에서 지원하는 연산자와 파이썬 라이브러리로 구현해야 한다.
2. **rknn 모델 변환**: rknn_toolkit2 라이브러리를 이용해, 구현한 트랜스포머 모델을 `.rknn` 모델로 변환한다.
3. **rknnpu에서 모델 실행**: rknn_toolkit_lite2 라이브러리를 이용해 변환된 rknn 모델을 npu 보드에서 실행할 수 있게 한다.

## 레포지토리 구조
<details>
<summary>Structure</summary>
<div markdown=1>

📦Transformer-to-RKNN  
 ┣ 📂ONNX  
 ┃ ┣ 📜convertONNX.py  
 ┃ ┣ 📜transformer.onnx  
 ┃ ┣ 📜transformer.rknn  
 ┃ ┗ 📜transformerONNX.py  
 ┣ 📂datasets  
 ┃ ┣ 📜load_dataset.py  
 ┃ ┣ 📜srcdataset.txt  
 ┃ ┗ 📜tgtdataset.txt  
 ┣ 📂pytorch  
 ┃ ┣ 📜convert.py  
 ┃ ┗ 📜transformer.py  
 ┣ 📂test  
 ┃ ┣ 📜convert.py  
 ┃ ┣ 📜load_datasets.py  
 ┃ ┣ 📜test.py  
 ┃ ┗ 📜transformer.py  
 ┗ 📜README.md  

</div>
</details>

## 코드 상세
1. [datasets](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/datasets)
    * `load_dataset.py`: 허깅페이스의 데이터셋을 불러오고, BertTokenizer를 사용해 데이터를 토크나이징해서 별개의 파일로 저장한다.
    * 현재는 WMT16 en-de 데이터셋을 샘플 데이터셋으로 사용하고 있다. (src:de, tgt:en)
    * 추후 데이터셋의 추가와 이에따른 파일명 수정이 필요해 보인다.
    
2. [ONNX](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/ONNX)
    * `transformerONNX.py`: 트랜스포머 모델 구현 및 저장. 
      * 데이터셋은 위에서 저장한 텍스트 파일을 이용한다.
      * 트랜스포머 모델 구현은 파이토치를 사용하되, `aten::unflatten`이나 `aten::scaled_dot_product_attention`과 같이 rknn_toolkit2에서 지원하지 않는 연산은 수학적으로 구현한다.
      * 구현된 모델의 저장은 ONNX로 이루어진다.
    * `convertONNX.py`: ONNX 모델을 RKNN모델로 변환.
      * 추후 인풋 사이즈와 데이터셋 사이즈를 동적으로 사용할 수 있도록 코드의 수정이 필요해 보인다.

3. [pytorch](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/pytorch), [test](https://github.com/ei-ai/Transformer-to-RKNN/tree/main/test)
    * 두 파일 모두 테스트/기록용 파일이다.
    * pytorch: 정상적으로 작동하지 않으나, 트랜스포머 구현 부분은 추후에 활용될 여지가 있어 남겨두었다.
    * test: 모델 구현이나 변환에 있어 여러가지 수정사항을 가한 파일들이다. 추후에 수정하여 기존 파일을 대체할 수 있어 해당 레포지토리에 추가해두었다.