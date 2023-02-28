import tensorflow as tf

from onnx_tf.backend import prepare
import onnx


onnx_model = onnx.load('./best.onnx')

tf_rep = prepare(onnx_model)

# to tensorflow
tf_rep.export_graph("./best")


converter = tf.lite.TFLiteConverter.from_saved_model('./best')

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
#
# #tflite int8은 데이터셋 필요해보임 아래 링크 참조
# https://www.tensorflow.org/lite/performance/post_training_integer_quant



tflite_model = converter.convert()
open("best.tflite", "wb").write(tflite_model)