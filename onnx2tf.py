import onnx
from onnx_tf.backend import prepare
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# onnx load
onnx_model = onnx.load('./onnx/best_res18_CAECAM_bn_256_for_test.onnx')
# onnx to tensorflow convert
tf_rep = prepare(onnx_model)
# tensorflow model save
tf_rep.export_graph("./tensorflow/best_res18_CAECAM_bn_256_for_test")
