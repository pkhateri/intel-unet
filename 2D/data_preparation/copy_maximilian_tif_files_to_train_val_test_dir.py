from glob import glob
from random import shuffle
import os, csv

"""
Inputs:
    - csv file including:
        - list of 2d images & their corresponding patient dir
        - format: "Kontrolle_01, img_id.tiff, bscan_num"
    - data_dir
    - output_dir
Steps:
    - read list of files from csv files
    - copy them to the corresponding train, val and test dir
"""
data_dir = '/usr/local/scratch/parisa/data/maximilian/OCT-Normal-Data/'
output_dir = '/usr/local/scratch/parisa/data/maximilian/intel_unet/tif/'


def read_csv_file(csv_filename):
    '''
    1. read csv file and store it in an array
    '''
    rows=[] # row1="Kontrolle_01, img_id.tiff, bscan_num"
    with open(csv_filename, 'r') as csv_f:
        csv_reader = csv.reader(csv_f)
        for row in csv_reader:
            rows.append(row[0])
    return rows

for s in ['train', 'val', 'test']:
    csv_file = os.path.join(output_dir, s+".csv")
    if not os.path.exists(os.path.join(output_dir, s)):
        os.mkdir(os.path.join(output_dir, s))
    rows = read_csv_file(csv_file)
    for row in rows:  # row1="Kontrolle_01, img_id.tiff, bscan_num"
        #/usr/local/scratch/parisa/data/maximilian/OCT-Normal-Data/Kontrolle_33/labels/Kontrolle_33_121_8EDA1F60_1024x496.tif
        label_img_path = os.path.join(data_dir, row.split(' ')[0], 'labels', row.split(' ')[0]+'_'+(row.split(' ')[2]).zfill(3)+'_'+row.split(' ')[1][:-4]+'_1024x496.tif')
        cmd = 'cp %s %s' %(label_img_path, os.path.join(output_dir, s))
        print(cmd)
        os.system(cmd)
