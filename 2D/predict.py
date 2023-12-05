#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
Takes a trained model and performs inference on a few validation examples.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

import numpy as np
import tensorflow as tf
import time
from tensorflow import keras as K
import settings
import argparse
from dataloader_2d import DatasetGenerator, get_2d_filelist

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


parser = argparse.ArgumentParser(
    description="TensorFlow Inference example for trained 2D U-Net model on BraTS.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the TensorFlow inference model filename")
parser.add_argument("--use_pconv",help="use partial convolution based padding",
                    action="store_true",
                    default=settings.USE_PCONV)
parser.add_argument("--output_pngs", default=settings.OUTPUT_PNGS,
                    help="the directory for the output prediction pngs")
parser.add_argument("--intraop_threads", default=settings.NUM_INTRA_THREADS,
                    type=int, help="Number of intra-op-parallelism threads")
parser.add_argument("--interop_threads", default=settings.NUM_INTER_THREADS,
                    type=int, help="Number of inter-op-parallelism threads")
parser.add_argument("--crop_dim", default=settings.CROP_DIM,
                    type=int, help="Crop dimension for images")
parser.add_argument("--seed", default=settings.SEED,
                    type=int, help="Random seed")
parser.add_argument("--split", type=float, default=settings.TRAIN_TEST_SPLIT,
                    help="Train/testing split for the data")
parser.add_argument("--batch_size", type=float, default=settings.BATCH_SIZE,
                    help="the batch size for training")
parser.add_argument("--input_type",
                    default=settings.INPUT_TYPE,
                    help="input image type: 2D or 3D")
parser.add_argument("--use_saved_model",
                    default=settings.USE_SAVED_MODEL,
                    help="start the model from pretrained weights",
                    action="store_true")
parser.add_argument("--saved_model_path",
                    help="path to the previously trained weights (saved model)",
                    default=os.path.join(settings.SAVED_MODEL_PATH))

args = parser.parse_args()

def test_intel_tensorflow():
    """
    Check if Intel version of TensorFlow is installed
    """
    import tensorflow as tf

    print("We are using Tensorflow version {}".format(tf.__version__))

    major_version = int(tf.__version__.split(".")[0])
    if major_version >= 2:
       from tensorflow.python import _pywrap_util_port
       print("Intel-optimizations (DNNL) enabled:", _pywrap_util_port.IsMklEnabled())
    else:
       print("Intel-optimizations (DNNL) enabled:", tf.pywrap_tensorflow.IsMklEnabled())

test_intel_tensorflow()

def plot_results(ds, batch_num, png_directory):

    plt.figure(figsize=(10,10))

    img, msk = next(ds.ds)

    # TODO to be decided
    #idx = np.argmax(np.sum(np.sum(msk[:,:,:,0], axis=1), axis=1)) # find the slice with the largest tumor
    idx = np.random.randint(0, args.batch_size-1) # randomly find a slice in the current batch

    plt.subplot(1, 3, 1)
    plt.imshow(img[idx, :, :, 0], cmap="bone") #, origin="upper") # comment "lower to avoid inverting
    plt.title("Image", fontsize=20)

    plt.subplot(1, 3, 2)
    plt.imshow(msk[idx, :, :], cmap="bone") #, origin="lower")
    plt.title("Ground truth", fontsize=20)

    plt.subplot(1, 3, 3)

    print("Index {}: ".format(idx), end="")

    # Predict using the TensorFlow model
    start_time = time.time()
    prediction = model.predict(img[[idx]])
    print("Elapsed time = {:.4f} msecs, ".format(1000.0*(time.time()-start_time)), end="")

    plt.imshow(prediction[0,:,:,0], cmap="bone") #, origin="lower")
    dice_coef = calc_dice(msk[idx], prediction)
    print("Dice coefficient = {:.4f}, ".format(dice_coef), end="")
    plt.title("Prediction\nDice = {:.4f}".format(dice_coef), fontsize=20)

    save_name = os.path.join(png_directory, "prediction_tf_{}_{}.png".format(batch_num, idx))
    print("Saved as: {}".format(save_name))
    plt.savefig(save_name)

if __name__ == "__main__":

    # Load model
    model_filename = os.path.join(args.saved_model_path)
    if args.use_pconv:
        from model_pconv import unet
        unet_model = unet(use_pconv=True)
    else:
        from model import unet
        unet_model = unet()
    model = unet_model.load_model(model_filename)

    # Create output directory for predictions
    output_dir = args.output_pngs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load all data in one batch and predict them one by one over a loop
    if args.input_type=='2D':
        from dataloader_2d import DatasetGenerator, get_2d_filelist
        __, __, testFiles = get_2d_filelist(data_path=args.data_path, seed=args.seed, split=args.split)
    ds_test_all = DatasetGenerator(testFiles, batch_size=len(testFiles), #read all at one batch
                                    crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)
    images, __ = next(ds_test_all.ds) # this is one batch which currently is the whole test data
    for i in range(len(images)):
        pred = model.predict(images[[i]])[0,:,:,0]
        plt.imshow(pred, cmap="bone") #, origin="lower")
        #plt.title("Prediction\nDice = {:.4f}".format(dice_coef), fontsize=20)
        output_filename = os.path.join(args.output_pngs, "prediction_{}.png".format(i))
        print("Saved as: {}".format(output_filename))
        plt.savefig(output_filename)
