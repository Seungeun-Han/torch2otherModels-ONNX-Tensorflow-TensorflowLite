import tensorflow as tf

from onnx_tf.backend import prepare
import onnx


onnx_model = onnx.load('./resnet50.onnx')

tf_rep = prepare(onnx_model)

# to tensorflow
tf_rep.export_graph("./resnet50")


converter = tf.lite.TFLiteConverter.from_saved_model('./resnet50')

#
# #tflite float32로 변환은 그냥 convert()만 하면 됨
#
#
# #tflite float16으로 변환
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
#
# #tflite dynamic으로 변환
# converter.optimizations = [tf.lite.Optimize.DEFAULT]



tflite_model = converter.convert()
open("resnet50.tflite", "wb").write(tflite_model)