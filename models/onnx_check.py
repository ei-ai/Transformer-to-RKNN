import onnx
from onnx import checker

model_path = "transformer.onnx"
try:
    model = onnx.load(model_path)
    checker.check_model(model)
    print("The ONNX model is valid.")
except Exception as e:
    print(f"Invalid ONNX model: {e}")
