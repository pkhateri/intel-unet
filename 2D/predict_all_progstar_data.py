#
# Parisa Khateri @ 08.12.2023
#

"""
reads all progsatr png files and predicts them with the pretrained model with healthy dataset
"""

import os, glob
import json


input_all_pngs_path ='/projects/progstar/cleaning_process_oct_progstar02/OCT_imgs_files/heyex_export_raw/renamed_vol/y49/png_dirs/'
output_all_pngs_path = '/usr/local/scratch/parisa/data/progstar/unet_predictions/'
input_type = '2D'
saved_model_path = "/projects/parisa/git_software/intel-unet/2D/output/20231201_2d_unet_maximilian_adam_lr0001_filter16_batchsize10_epoch100_earlystop12_augmentationFalse"
for data_path in glob.glob(input_all_pngs_path+"1*z496_x1024"):
    print("###################################################################################")
    print(data_path.split('/')[-1])
    output_pngs = os.path.join(output_all_pngs_path, data_path.split('/')[-1])
    if not os.path.exists(output_pngs):
        os.mkdir(output_pngs)
    cmd = "python3 predict.py --data_path {} --output_pngs {} --input_type {} --saved_model_path {}".format(data_path, output_pngs, input_type, saved_model_path)
    #print(cmd)
    os.system(cmd)
