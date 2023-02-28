import numpy as np
import torch
from networks.dml_csr_danet_res18_bn import DML_CSR
import tensorflow as tf
import cv2

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# torch model pth
TORCH_PATH = 'best.pth'
# TF model pth
PB_PATH = './best'
batch_size = 1

# 모델에 대한 입력값
x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
image = cv2.imread('./00001.jpg', cv2.IMREAD_COLOR) # shape 256, 256, 3
# 이미지 전처리
image = image.transpose(2,1,0) # CHW  -> HWC # 3 256 256
image = np.expand_dims(image, axis=0) # 1 3 256 256

# torch
# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None

# model load
network = DML_CSR(20, trained=False)
network.load_state_dict(torch.load(TORCH_PATH, map_location=map_location))

# 모델을 추론 모드로 전환합니다
network.eval()

torch_image = torch.Tensor(image)
torch_out = network(torch_image) # torch output

# tensorflow
image = tf.cast(image, tf.float32) # 전처리 for pb
model = tf.saved_model.load(PB_PATH)
model = model.signatures["serving_default"]
results = model(image)
parsing = results['output_0'] # tensorflow output
# print(parsing)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
np.testing.assert_allclose(to_numpy(torch_out), parsing, rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")