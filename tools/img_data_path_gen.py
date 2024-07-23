import glob
import os

IMG_DATA_DIR_PATH = './test_image/img_data'
IMG_DATA_TXT_SAVE_PATH = './test_image/img_data.txt'

# Get file path list
# img_file_path_list = glob.glob('../dataset/train/img_data/*.jpg')
# print(img_file_path_list)

# Get file name list
img_file_name_list = sorted(os.listdir(IMG_DATA_DIR_PATH))
# print(img_file_name_list)

# Generate txt file for training data list
f = open(IMG_DATA_TXT_SAVE_PATH, 'w')
for img_file_name in img_file_name_list:
    f.write(img_file_name + "\n")
f.close()