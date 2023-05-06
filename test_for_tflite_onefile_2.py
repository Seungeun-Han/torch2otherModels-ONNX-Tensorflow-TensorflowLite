#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import time

import tensorflow as tf
import argparse
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# from tensorflow import keras
import time
from copy import deepcopy
#from inplace_abn import InPlaceABN
from datasets import datasets_necklace_last
#from networks import dml_csr
from utils import miou_necklaceLast

torch.multiprocessing.set_start_method("spawn", force=True)

DATA_DIRECTORY = 'C:/Users/USER/Downloads/TFLiteRetinaFace-main/TFLiteRetinaFace/app/src/main/assets/00004_kf94.png'
SAVE_DIRECTORY = './TFLite/best_res18_CACAM_4colors_112_bn_noMaxP_interp_argmax_float32'
IGNORE_LABEL = 255
NUM_CLASSES = 20  # 20
TFLITE_PATH = "./TFLite/best_res18_CACAM_4colors_112_bn_noMaxP_interp_argmax_float32.tflite"
INPUT_SIZE = [112, 112]


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DML_CSR Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC datasets.")
    parser.add_argument("--out-dir", type=str, default=SAVE_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC datasets.")
    parser.add_argument("--dataset", type=str, default='test',
                        help="Path to the file listing the images in the datasets.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    #parser.add_argument("--restore-from", type=str, default='./snapshots/best.pth',
                        #help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="choose gpu numbers") 
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument("--model_type", type=int, default=0,
                        help="choose model type") 
    return parser.parse_args()

def one_image_test(interpreter, input_size):

    height = input_size[0]
    width = input_size[1]
    image = cv2.imread("C:/Users/USER/Downloads/TFLiteRetinaFace-main/TFLiteRetinaFace/app/src/main/assets/00004_kf94_112.png", cv2.IMREAD_COLOR)

    # BGR2RBG
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(image)
    # rgb = cv2.merge((r, g, b))
    # print(rgb[0][0])
    # print(rgb.shape)

    # 전처리
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    image = transform(image)

    image = image.transpose(2, 0)  # CHW  -> HWC
    image = np.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)

    print(image.shape)
    # print(model.structured_outputs)  -> {'output_0': TensorSpec(shape=(None, 20, 64, 64), dtype=tf.float32, name='output_0')}
    # results = model(image)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']

    print(input_shape)
    print(output_shape)


    s = time.time()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    curr_time = (time.time()-s)
    print(curr_time)
    # print(output_data)
    # parsing = output_data['output_0'].numpy()
    parsing = output_data
    print(parsing.shape)
    # parsing = results['output_0'].numpy()
    # parsing = cv2.resize(parsing, (height, width))
    # interp = tf.keras.layers.UpSampling2D(size=(height, width), interpolation='bilinear')

    # parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
    # interp = torch.nn.Upsample(size=(height, width), mode='bilinear', align_corners=True)
    # parsing = interp(torch.tensor(parsing)).data.cpu().numpy()

    parsing_preds = np.asarray(parsing, dtype=np.uint8)

    # interp만 했을 때
    # parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
    # parsing_preds = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

    cv2.imshow("parsing", parsing_preds[0]*10)
    cv2.waitKey()

    return parsing_preds

def valid_for_test(interpreter, valloader, input_size, num_samples, dir=None, dir_edge=None, dir_img=None):
    height = input_size[0]
    width = input_size[1]

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, \
        record_shapes=False, profile_memory=False) as prof:
        #model.eval()
        parsing_preds = np.zeros((num_samples, height, width), dtype=np.uint8)
        scales = np.zeros((num_samples, 2), dtype=np.float32)
        centers = np.zeros((num_samples, 2), dtype=np.int32)

        idx = 0
        interp = torch.nn.Upsample(size=(height, width), mode='bilinear', align_corners=True)

        with torch.no_grad():
            for index, batch in enumerate(valloader):
                image, meta = batch
                num_images = image.size(0)
                if index % 10 == 0:
                    print('%d  processd' % (index * num_images))

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                scales[idx:idx + num_images, :] = s[:, :]
                centers[idx:idx + num_images, :] = c[:, :]

                image = image.transpose(3, 1)  # CHW  -> HWC
                # print(image.shape)
                image = tf.cast(image, tf.float32)
                # print(model.structured_outputs)  -> {'output_0': TensorSpec(shape=(None, 20, 64, 64), dtype=tf.float32, name='output_0')}
                # results = model(image)

                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                # Test model on random input data.
                interpreter.set_tensor(input_details[0]['index'], image)

                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                # print(output_data)
                # parsing = output_data['output_0'].numpy()
                parsing = output_data
                print(parsing.shape)

                # parsing = interp(torch.tensor(parsing)).data.cpu().numpy()
                # parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                # parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(parsing, dtype=np.uint8)
                idx += num_images

                if dir is not None:
                    for i in range(len(meta['name'])):
                        # cv2.imwrite(os.path.join(dir, meta['name'][i] + '.png'), np.asarray(np.argmax(parsing, axis=3))[i])
                        cv2.imwrite(os.path.join(dir, meta['name'][i] + '.png'), parsing[0]*10)

        parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    # print(args.gpu)

    cudnn.benchmark = True
    cudnn.enabled = True

    # model = onnx.load(ONNX_PATH)
    # model = dml_csr.DML_CSR(args.num_classes, InPlaceABN, False)
    # model = tf.saved_model.load(PB_PATH)
    # model = model.signatures["serving_default"]
    #print(f(x=tf.constant([[1.]])))
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(TFLITE_PATH)
    interpreter.allocate_tensors()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # dataset = datasets_necklace_last.FaceDataSet(args.data_dir, args.dataset, crop_size=INPUT_SIZE, transform=transform)
    # num_samples = len(dataset)

    # valloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    #restore_from = args.restore_from
    #print(restore_from)
    #state_dict = torch.load(restore_from, map_location='cuda:0')
    #model.load_state_dict(state_dict)

    #model.cuda()
    #model.eval()

    save_path = os.path.join(args.out_dir, args.dataset, 'parsing')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    one_image_test(interpreter, INPUT_SIZE)
    # parsing_preds, scales, centers = valid_for_test(interpreter, valloader, INPUT_SIZE, num_samples, save_path)
    # mIoU, f1 = miou_necklaceLast.compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, INPUT_SIZE, args.dataset, reverse=True)
    #
    # print(mIoU)
    # print(f1)

if __name__ == '__main__':
    main()
