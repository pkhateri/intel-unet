from glob import glob
from random import shuffle
import os
import xml.etree.ElementTree as ET

"""
Inputs:
    - data_dir
    - output_dir
Outputs;
    - csv files for train, test and val in output dir
Steps:
    - create a list of patients
    - shuffle and divide the data
    - create list of file names and write in train.csv, val.csv, test.csv
    - csv files are formatted as: Kontrolle_01, img_id.tiff, bscan_num
"""
data_dir = '/projects/parisa/data/maximilian/OCT-Normal-Data/'
output_dir = '/projects/parisa/data/maximilian/intel_unet/json/'
train_csv_file = os.path.join(output_dir, "train.csv")
val_csv_file = os.path.join(output_dir, "val.csv")
test_csv_file = os.path.join(output_dir, "test.csv")

"""
creat a list of patients
"""
patient_list = []
for d in glob(data_dir+"Kontrolle_*"):
    patient_list.append(d.split('/')[6])

"""
    divide the data
    train/va/test: 8/1/1
    randomly select patients
"""
shuffle(patient_list)
n = len(patient_list)
n_train = round(0.8*n)
n_val = round((n - n_train)/2)
n_test = n - n_train - n_val
train_patient_list = []
val_patient_list = []
test_patient_list = []
for i, p in enumerate(patient_list):
    if i<n_train: train_patient_list.append(p)
    elif i<n_train+n_val: val_patient_list.append(p)
    else: test_patient_list.append(p)

print("train_patient_list", train_patient_list)
print("val_patient_list", val_patient_list)
print("test_patient_list", test_patient_list)


"""
write csv files
"""
with open(train_csv_file, 'w') as f:
    for p in train_patient_list:
        patient_id = p
        xml_file = glob(data_dir+p+"/*.xml")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = root.find('BODY/Patient/Study/Series')
        images = data.findall('Image')
        for i in images:
            if int(i.find('ID').text) > 0:
                bscan_num = int(i.find('ID').text)
                img_id = os.path.basename(i.find('ImageData/ExamURL').text.split('\\')[5])
                print(patient_id, img_id, bscan_num, file=f)
with open(val_csv_file, 'w') as f:
    for p in val_patient_list:
        patient_id = p
        xml_file = glob(data_dir+p+"/*.xml")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = root.find('BODY/Patient/Study/Series')
        images = data.findall('Image')
        for i in images:
            if int(i.find('ID').text) > 0:
                bscan_num = int(i.find('ID').text)
                img_id = os.path.basename(i.find('ImageData/ExamURL').text.split('\\')[5])
                print(patient_id, img_id, bscan_num, file=f)
with open(test_csv_file, 'w') as f:
    for p in test_patient_list:
        patient_id = p
        xml_file = glob(data_dir+p+"/*.xml")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = root.find('BODY/Patient/Study/Series')
        images = data.findall('Image')
        for i in images:
            if int(i.find('ID').text) > 0:
                bscan_num = int(i.find('ID').text)
                img_id = os.path.basename(i.find('ImageData/ExamURL').text.split('\\')[5])
                print(patient_id, img_id, bscan_num, file=f)
