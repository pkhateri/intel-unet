"""
Inputs:
    csv file including list of 2d images
    img_dir where the 2d images are located
    label_dir where the *_Surfaces_Iowa.xml files are located
    output_dir where json files will be located
Outputs:
    json files for individual 2d images with the img_path, label_path and bscan_num_for_png
"""

import os, sys, getopt, csv, json


def read_csv_file(csv_filename):
    '''
    1. read csv file and store it in an array -> rows
    '''
    rows=[]
    with open(csv_filename, 'r') as csv_f:
        csv_reader = csv.reader(csv_f)
        for row in csv_reader:
            if row[0].startswith('1'):
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
    output_json_file = os.path.basename(img_path)[:-4] + ".json"
    with open(os.path.join(output_dir, output_json_file), 'w') as outfile:
        json.dump(writedata, outfile)


def main(argv):
    '''
    default inputs:
    '''
    csv_file = '' #'train.csv'
    output_dir = '' #'/projec/parisa/data/progstar/intel_unet/train/'
    img_dir = '' #'/projects/progstar/all_oct_imgs_progstar02_quality_checked/adequate/'
    label_dir = '' #'/usr/local/scratch/iowa_segmentation/'

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
        img_path = os.path.join(img_dir, row)
        center_id = row[:2]
        label_path = os.path.join(label_dir,'center'+str(center_id),row[:-12],str(row[:-12])+'_Surfaces_Iowa.xml')
        bscan_num_for_png = int((row[-7:])[:3]) # iowa and png have different conventions for bscan numbers (n_png = 48 - n_iowa)
        write_json_file(output_dir, img_path, label_path, bscan_num_for_png)


if __name__ == "__main__":
    main(sys.argv[1:])
