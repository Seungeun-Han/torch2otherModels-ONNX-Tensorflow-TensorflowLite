import onnx
from onnx_tf.backend import prepare

Name_of_ONNX = 'resnet50.onnx'
Name_of_TF = 'resnet50'

# onnx load
onnx_model = onnx.load(Name_of_ONNX)

# onnx to tensorflow convert
tf_rep = prepare(onnx_model)

# tensorflow model save
tf_rep.export_graph(Name_of_TF)
