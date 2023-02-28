import tensorflow as tf

from onnx_tf.backend import prepare
import onnx


onnx_model = onnx.load('./onnx/best_resnet50_CAEtriCAMask_256_for_test.onnx')

tf_rep = prepare(onnx_model)
print("준비끝")
tf_rep.export_graph("./tensorflow/best_resnet50_CAEtriCAMask_256_for_test")
print("pb변환끝")

converter = tf.lite.TFLiteConverter.from_saved_model('./tensorflow/best_resnet50_CAEtriCAMask_256_for_test')

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
open("./TFLite/best_resnet50_CAEtriCAMask_256_for_test.tflite", "wb").write(tflite_model)