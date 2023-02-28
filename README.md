# torch2otherModels(ONNX, Tensorflow, TensorflowLite)
 
This repository can be used to convert PyTorch to other types of model such as ONNX, Tensorflow and TensorflowLite.

If you want to convert pytorch to TFLite, to utilize the pretrained Deep Learning model in Android, you need to follow this step:

1. Pytorch -> ONNX
2. ONNX -> Tensorflow
3. Tensorflow -> TensorflowLite

To illustrate how to use this code, consider converting ResNet18 (PyTorch Model) as an example.

## How to use?

### 1. Pytorch -> ONNX
To convert PyTorch to ONNX, run __pytorch2onnx.py__.

If converting has no problem, you can see "The model is valid!".


#### Customize

If you want to convert your model, you have to import your model and set the appropriate shape of input.


### 2. ONNX -> Tensorflow
Run __onnx2tf.py__.


#### Customize

Modify the file path or name(Name_of_ONNX, Name_of_TF)


### 3. Tensorflow -> TensorflowLite
Run __tf2TFLite.py__.

#### Customize

Modify the file path or name(TF_PATH, TFLite_name)
