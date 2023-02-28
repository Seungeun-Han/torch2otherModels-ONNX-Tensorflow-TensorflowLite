import numpy as np
import torch.onnx
from torchvision.models import resnet50
import onnx
import onnxruntime
import time

def to_numpy(tensor): # tensor to numpy
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# pytorch model
network = resnet50(pretrained=True)
network.eval()

# Set appropriate Input shape
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

# convert to onnx
torch.onnx.export(network,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "resnet50.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=12,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

# 변환된 onnx 모델 불러오기
onnx_model = onnx.load('resnet50.onnx')

# 모델 valid한지 확인
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s" % e)
else:
    print("The model is valid!")

# onnx 모델 실행을 위한 inference session
ort_session = onnxruntime.InferenceSession("resnet50.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# ONNX input
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# s = time.time()

# ONNX 런타임에서 계산된 결과값
ort_outs = ort_session.run(None, ort_inputs)

# print('onnx inference time:', time.time()-s)
