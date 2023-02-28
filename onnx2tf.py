import onnx
from onnx_tf.backend import prepare

# onnx load
onnx_model = onnx.load('best.onnx')
# onnx to tensorflow convert
tf_rep = prepare(onnx_model)
# tensorflow model save
tf_rep.export_graph("best")
