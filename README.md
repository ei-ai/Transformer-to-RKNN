# Transformer-to-RKNN

`convert.py` 계속 오류생기는데 어떻게 처리해야 할 지 모르겠음 일단 수정한 데까지 올림
<details>
<summary>에러메세지</summary>
<div markdown=1>

--> Loading PyTorch model
W load_pytorch: Catch exception when torch.jit.load:
    RuntimeError('PytorchStreamReader failed locating file constants.pkl: file not found')
W load_pytorch: Make sure that the torch version of './transformer_scripted.pt' is consistent with the installed torch version '2.0.1'!

E load_pytorch: Traceback (most recent call last):
E load_pytorch:   File "rknn/api/rknn_base.py", line 1572, in rknn.api.rknn_base.RKNNBase.load_pytorch
E load_pytorch:   File "/home/ubuntu/miniconda3/envs/rknn2/lib/python3.8/site-packages/torch/jit/_serialization.py", line 162, in load
E load_pytorch:     cpp_module = torch._C.import_ir_module(cu, str(f), map_location, _extra_files, _restore_shapes)  # type: ignore[call-arg]
E load_pytorch: RuntimeError: PytorchStreamReader failed locating file constants.pkl: file not found

W If you can't handle this error, please try updating to the latest version of the toolkit2 and runtime from:
  https://console.zbox.filez.com/l/I00fc3 (Pwd: rknn)  Path: RKNPU2_SDK / 2.X.X / develop /
  If the error still exists in the latest version, please collect the corresponding error logs and the model,
  convert script, and input data that can reproduce the problem, and then submit an issue on:
  https://redmine.rock-chips.com (Please consult our sales or FAE for the redmine account)
Load model failed!

  
</div>
  
</details>
