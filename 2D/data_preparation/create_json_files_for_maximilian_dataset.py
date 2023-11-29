"""
Inputs:
    csv file including list of 2d images with their corresponding patient dir, e.g. row1="Kontrolle_01, img_id.tiff, bscan_num"
    img_dir where the 2d images are located
    label_dir where the *_Surfaces_Iowa.xml files are located
    output_dir where json files will be located
Outputs:
    json files for individual 2d images with the img_path, label_path and bscan_num_for_png
"""

import os, sys, getopt, csv, json

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


def write_json_file(output_dir, img_path, label_path, bscan_num_for_png):
    '''
    create a dictionary
    '''
    writedata = {}
    writedata["imagePath"] = img_path
    writedata["labelPath"] = label_path
    writedata["bScanNumForPNG"] = bscan_num_for_png # iowa and png have different conventions for bscan numbers (n_png = 48 - n_iowa)

    '''
    dump the dictionary to the output file
    '''
    output_json_file = img_path.split("/")[-2]+'_'+str(bscan_num_for_png).zfill(3)+'_'+os.path.basename(img_path)[:-4]+".json"
    print(output_json_file)
    with open(os.path.join(output_dir, output_json_file), 'w') as outfile:
        json.dump(writedata, outfile)

def main(argv):
    '''
    inputs:
    '''
    csv_file = '' #'/projects/parisa/data/maximilian/intel_unet/train.csv' Kontrolle_01, img_id.tiff, bscan_num
    output_dir = '' #'/projects/parisa/data/maximilian/intel_unet/train/'
    img_dir = '' #'/projects/parisa/data/maximilian/OCT-Normal-Data/'
    label_dir = '' #'/projects/parisa/data/maximilian/OCT-Normal-Data/iowa_format/'

    opts, args = getopt.getopt(argv,"hc:o:i:l:",["csv_file=","output_dir=", "img_dir=", "label_dir="])
    for opt, arg in opts:
        if opt == '-h':
            print ('Usage: python create_json_files_for_iowa_output.py -c <csv_file_path> -o <output_dir>\
            -i <img_dir> -l <label_dir>')
            sys.exit()
        elif opt in ("-c", "--csv_file"):
            csv_file = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-i", "--img_dir"):
            img_dir = arg
        elif opt in ("-l", "--label_dir"):
            label_dir = arg

    if len(argv) < 5:
        sys.exit('Usage: python create_json_files_for_iowa_output.py -c <csv_file_path> -o <output_dir>\
        -i <img_dir> -l <label_dir>')

    if not os.path.exists(csv_file):
        sys.exit("ERROR: csv_file {} does not exist!".format(csv_file))
    if not os.path.exists(output_dir):
        sys.exit("ERROR: output_dir {} does not exist!".format(output_dir))
    if not os.path.exists(img_dir):
        sys.exit("ERROR: img_dir {} does not exist!".format(img_dir))
    if not os.path.exists(label_dir):
        sys.exit("ERROR: label_dir {} does not exist!".format(label_dir))

    rows = read_csv_file(csv_file)
    for row in rows:
        img_path = os.path.join(img_dir, row.split(' ')[0], row.split(' ')[1])
        label_path = os.path.join(label_dir, row.split(' ')[0], row.split(' ')[0]+'_Surfaces_Iowa.xml')
        bscan_num_for_png = int(row.split(' ')[2]) # iowa and tiff have different conventions for bscan numbers (n_png = n_tot - n_iowa)
        write_json_file(output_dir, img_path, label_path, bscan_num_for_png)


if __name__ == "__main__":
    main(sys.argv[1:])
