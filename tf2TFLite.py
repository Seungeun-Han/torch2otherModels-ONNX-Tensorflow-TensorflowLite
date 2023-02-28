import tensorflow as tf

TF_PATH = './best'
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH) # path to the SavedModel directory
# converter.target_spec.supported_ops = [ # 없으면 에러뜸
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
# Quantization Option
# 옵션 정해주지 않으면 float32로 변환됨
# line14 옵션 설정하면 int8로 변환됨
# line14,15 옵션 설정하면 float16으로 변환됨
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

# converter.experimental_supported_backends = ["GPU"]

# tensorflow to TFLite convert # 변환 시간 오래걸림
tflite_model = converter.convert()

# TFLite model save
with open('./best', 'wb') as f:
  f.write(tflite_model)