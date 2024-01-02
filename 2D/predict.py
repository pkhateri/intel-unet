#
# Parisa Khateri @ 07.12.2023
#

"""
Takes a trained model and performs prediction on the given input images
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

import numpy as np
import tensorflow as tf
import time
from tensorflow import keras as K
import settings
import argparse
from dataloader_2d import DatasetGenerator
from PIL import Image
import json

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


parser = argparse.ArgumentParser(
    description="TensorFlow Inference example for trained 2D U-Net model on BraTS.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--use_pconv",help="use partial convolution based padding",
                    action="store_true",
                    default=settings.USE_PCONV)
parser.add_argument("--output_pngs", default=settings.OUTPUT_PNGS,
                    help="the directory for the output prediction pngs")
parser.add_argument("--crop_dim", default=settings.CROP_DIM,
                    type=int, help="Crop dimension for images")
parser.add_argument("--seed", default=settings.SEED,
                    type=int, help="Random seed")
parser.add_argument("--split", type=float, default=settings.TRAIN_TEST_SPLIT,
                    help="Train/testing split for the data")
parser.add_argument("--input_type",
                    default=settings.INPUT_TYPE,
                    help="input image type: 2D or 3D")
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

def get_2d_filelist(data_path):
    try:
        if os.path.isdir(data_path):
            img_files = [os.path.join(data_path, f)
                for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    except IOError as e:
        raise Exception("Folder {} doesn't exist".format(data_path))
    # Print information about the loaded data
    print("*" * 30)
    print("=" * 30)
    print("Number of files loaded  = {}".format(len(img_files)))
    print("=" * 30)
    print("*" * 30)
    return img_files

def get_dataset(filenames, batch_size):
    """
    Return a dataset
    """
    ds = self.generate_batch_from_files(filenames, batch_size)

    return ds

def generate_batch_from_files(filenames, batch_size):
    """
    Python generator which goes through a list of filenames to load.
    The files are 2D image. We yield them as a batch of 2D slices.
    This generator keeps yielding a batch of 2D slices at a time until all 2D images are loaded.

    idx: to track the number of 2d images which have been loaded so far. idx[0,num_filenames]
    idz: index for adding 2d images (files) to the batch. idz[0,batch_size]
    """

    idx = 0
    filename_batch = []
    while True:
        """
        Pack N_IMAGES files at a time to queue
        """
        for idz in range(batch_size): #loop to fill a batch with 2d images from png files

            filename = filenames[idx]
            if filename.endswith(".png") or filename.endswith(".tif"):
                img_and_label = np.array(Image.open(filename).convert("RGB"), dtype=np.float32)/255

                img = img_and_label[:,:,1]  # the green channel is the image
                img = np.expand_dims(img, axis=-1) # to be abale to concatenate later
                img = self.preprocess_img(img)

                label = img_and_label[:,:,0] # the red channel is the label
                label = np.expand_dims(label, axis=-1) # to be abale to concatenate later
                label = self.preprocess_label(label)
            elif filename.endswith(".json"):
                try:
                    with open(filename, 'r') as jsn:
                        dict = json.load(jsn)
                        img_path = dict["imagePath"]
                        label_path = dict["labelPath"]
                        bscan_num_for_png = int(dict["bScanNumForPNG"])
                        rgb_img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)/255

                        img = rgb_img[:,:,1] # all channels are the same
                        img = np.expand_dims(img, axis=-1) # to be abale to concatenate later
                        img = self.preprocess_img(img)

                        label = get_label_from_iowa_xml_file(label_path, bscan_num_for_png)
                        label = np.expand_dims(label, axis=-1) # to be abale to concatenate later
                        label = self.preprocess_label(label)

                except json.JSONDecodeError:
                    print(f"{jsn} is not a valid JSON file.")
            else:
                os.system.exit("Error: strange input file!", filename)

            if idz == 0:
                img_stack = img
                label_stack = label
            else:
                img_stack = np.concatenate((img_stack,img), axis=self.slice_dim)
                label_stack = np.concatenate((label_stack,label), axis=self.slice_dim)

            filename_batch.append(filename)

            idx += 1
            if idx >= len(filenames):
                idx = 0
                np.random.shuffle(filenames) # Shuffle the filenames for the next iteration

        # outside "for" loop, inside "while" loop
        img_batch = img_stack
        label_batch = label_stack

        if len(np.shape(img_batch)) == 3:
            img_batch = np.expand_dims(img_batch, axis=-1)
        if len(np.shape(label_batch)) == 3:
            label_batch = np.expand_dims(label_batch, axis=-1)

        # permute the dimension, i.e. bring channel to first position
        yield np.transpose(img_batch, [2,0,1,3]).astype(np.float32), np.transpose(label_batch, [2,0,1,3]).astype(np.float32), filename_batch

def get_boundaries_from_mask(mask):
    """
    input: mask is a numpy array with binary values (0,1)
    outputs:
         - 2 arrays of size=n_columns, each element representing y index of the boundary position
         - a 2d numpy array representing the boundary img with binary values
    """
    n_rows = mask.shape[0] # number of rows
    n_columns = mask.shape[1] # number of columns
    boundary_1 = np.zeros((n_columns))
    boundary_2 = np.zeros((n_columns))
    boundary_img = np.zeros((n_rows, n_columns))
    for j in range(n_columns):
        for i in range(n_rows):
            if i>0 and mask[i-1][j]==0 and mask[i][j]==1:
                boundary_1[j]=i
                boundary_img[i][j]=1
            if i<n_rows-1 and mask[i+1][j]==0 and mask[i][j]==1:
                boundary_2[j]=i
                boundary_img[i][j]=1
    return boundary_1, boundary_2, boundary_img

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
        from dataloader_2d import DatasetGenerator
        test_files = get_2d_filelist(data_path=args.data_path)
        ds_test = get_dataset(test_files, batch_size=len(testFiles), #read all at one batch
                              crop_dim=[args.crop_dim,args.crop_dim])
        images, __, filenames = next(ds_test.ds) # this is one batch which currently is the whole test data
    for i in range(len(images)):
        filename = os.path.basename(filenames[i])
        img = images[i]
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        img_255 = (img*255).astype(np.uint8)
        print("Predicting: {}".format(filename))
        pred_arr = model.predict(images[[i]])[0,:,:,0]
        #mid_range = pred[(pred>0.2)*(pred<0.8)]
        #mid_range_hist = np.hstack(mid_range)
        #_ = plt.hist(mid_range_hist, bins=20)
        pred_arr_255 = (pred_arr* 255).astype(np.uint8)
        pred_img = Image.fromarray(np.dstack([pred_arr_255, pred_arr_255, pred_arr_255]))
        #pred_filename = os.path.join(args.output_pngs, "prediction_{}.tif".format(filename[:-4]))
        #pred_img.save(pred_filename)

        thrsh = 0.5
        mask_arr = (pred_arr>thrsh).astype(int)
        mask_arr_255 = (mask_arr* 255).astype(np.uint8)
        mask_img = Image.fromarray(np.dstack([mask_arr_255, mask_arr_255, mask_arr_255]))
        mask_filename = os.path.join(args.output_pngs, "mask_{}.tif".format(filename[:-4]))
        mask_img.save(mask_filename)

        boundary_1, boundary_2, boundary_arr = get_boundaries_from_mask(mask_arr)
        boundary_arr_255 = (boundary_arr* 255).astype(np.uint8)
        boundary_img = Image.fromarray(np.dstack([boundary_arr_255,img_255,img_255])).convert("RGB")
        boundary_filename = os.path.join(args.output_pngs, "boundary_{}.tif".format(filename[:-4]))
        boundary_img.save(boundary_filename)

        output_dict = {
                        "img": filename,
                        "ILM": boundary_1.tolist(),
                        "Choroid": boundary_2.tolist(),
                      }
        json_filename = os.path.join(args.output_pngs, "{}.json".format(filename[:-4]))
        with open(json_filename, 'w') as json_file:
            json.dump(output_dict, json_file)

        #plt.plot(boundary_1, '-r')
        #plt.plot(boundary_2, '-b')
        #plot_name = os.path.join(args.output_pngs, "plot_{}.png".format(filename[:-4]))
        #plt.savefig(plot_name)
        #plt.close()
