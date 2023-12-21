import sys
# sys.path.append('/Users/yerin/Desktop/my_research/src/modules')
from pathlib import Path
# path
modules_path = Path(__file__).parent
src_path = modules_path.parent

sys.path.append(src_path)
import img_dataset
import os
import matplotlib.pyplot as plt
import pandas as pd

# img_dataset.img_data('/Users/yerin/Desktop/my_research/src/images/heatmap_test/', 'img1.jpeg')
# print(img_dataset.crop_size)

# initialize list
height_list = []
width_list = []
table_list = []

# get image directory
# dir_path = '/Users/yerin/Desktop/my_research/src/images/heatmap_test'

dir_path = str(src_path) + '/images/cat'
file_list = os.listdir(dir_path)
# print(file_list)
# sorted
# file_list = sorted(file_list, key=lambda x: int(x.split('img')[1].split('.jpeg')[0]))

def make_table():

    # get all images
    for img_file in file_list:
        _, crop_size, img_flatten, xy_flatten, _ = img_dataset.img_data(dir_path + '/', img_file)

        img_data = {
            'img_file' : img_file,
            'img_flatten': img_flatten,
            'xy_flatten' : xy_flatten,
        }

        table_list.append(img_data)

    # create table
    img_df = pd.DataFrame(table_list)
    img_df.index += 1 

    return img_df, crop_size
# img_df, _ = make_table()
# print(img_df['img_flatten'])




    


