import os
from sklearn.model_selection import train_test_split
import shutil

# 获取目录下所有文件名
def find_all_file(dir):
    for root, ds, fs in os.walk(dir):
        for f in fs:
            filename = f.split('.')[0]
            yield filename


if __name__ == '__main__':
    images_path = './train/images_enhancement/'
    labels_path = './train/labels/'
    yolo_data_images_path = './train/yolo_data/images/'
    yolo_data_labels_path = './train/yolo_data/labels/'

    filenames = list(find_all_file(images_path))
    filenames_train, filenames_test = train_test_split(filenames, test_size=0.1, random_state=0)
    filenames_train, filenames_val = train_test_split(filenames_train, test_size=0.111, random_state=0)

    filenames_3 = [filenames_train, filenames_test, filenames_val]
    destinations = ['train/', 'val/', 'test/']

    for i in range(3):
        filenames = filenames_3[i]
        images_src = images_path
        labels_src = labels_path
        images_des = yolo_data_images_path + destinations[i]
        labels_des = yolo_data_labels_path + destinations[i]
        for filename in filenames:
            src_img = images_src + filename + '.jpg'
            des_img = images_des + filename + '.jpg'
            src_label = labels_src + filename + '.txt'
            des_label = labels_des + filename + '.txt'
            shutil.copyfile(src_img, des_img)
            shutil.copyfile(src_label, des_label)
    print('copy finished!')

