import os
import cv2

'''
数据处理
将原始数据格式，调整yolo格式
原始格式：000010501024588.jpg 非金属打火机 805 365 965 436
原始格式：image class x1 y1 x2 y2
yolo格式: class x_center y_center width height
'''


# 归一化
def normalized(img_width, img_height, x1, y1, x2, y2):
    dw = 1. / img_width
    dh = 1. / img_height

    width = x2 - x1
    height = y2 - y1
    x_center = x2 - width / 2.
    y_center = y2 - height / 2.

    x_center = round(x_center * dw, 6)
    y_center = round(y_center * dh, 6)
    width = round(width * dw, 6)
    height = round(height * dh, 6)

    return x_center, y_center, width, height


# 反归一化
def re_normalized(img_width, img_height, x_center, y_center, width, height):
    dw = 1. / img_width
    dh = 1. / img_height

    x_center = x_center / dw
    y_center = y_center / dh
    width = width / dw
    height = height / dh

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return x1, y1, x2, y2


def find_all_file(dir):
    for root, ds, fs in os.walk(dir):
        for f in fs:
            filename = f.split('.')[0]
            yield filename


def point_to_yolo():
    images_dir = './train/images/'
    origin_txt_dir = './train/annotations/'
    labels_txt_dir = './train/labels/'
    for filename in find_all_file(origin_txt_dir):
        img_path = images_dir + filename + '.jpg'
        txt_path = origin_txt_dir + filename + '.txt'
        label_path = labels_txt_dir + filename + '.txt'

        image = cv2.imread(img_path)
        img_width = image.shape[1]
        img_height = image.shape[0]

        label_lines = []
        with open(txt_path, 'r') as f:
            data_lines = f.readlines()
            for data in data_lines:
                columns = data.split(' ')
                x1, y1, x2, y2 = tuple(list(map(float, columns[2:])))
                x_center, y_center, width, height = normalized(img_width, img_height, x1, y1, x2, y2)
                # print(x_center, y_center, width, height)
                label_lines.append('0 %s %s %s %s\n' % (x_center, y_center, width, height))

        with open(label_path, 'x') as f:
            for line in label_lines:
                f.write(line)


def yolo_to_point():
    yolo_images_dir = '../yolov5/runs/detect/exp10/'
    yolo_txt_dir = f'{yolo_images_dir}/labels/'  # class x_center y_center width height conf
    result_txt = './result.txt'

    result_txt_lines = []
    for filename in find_all_file(yolo_txt_dir):
        img_path = yolo_images_dir + filename + '.jpg'
        txt_path = yolo_txt_dir + filename + '.txt'

        image = cv2.imread(img_path)
        img_width = image.shape[1]
        img_height = image.shape[0]

        with open(txt_path, 'r') as f:
            data_lines = f.readlines()
            for data in data_lines:
                columns = data.split(' ')
                x_center, y_center, width, height, conf = tuple(list(map(float, columns[1:])))
                # if conf < 0.01:
                #     continue
                x1, y1, x2, y2 = re_normalized(img_width, img_height, x_center, y_center, width, height)
                # print(x_center, y_center, width, height)

                result_txt_lines.append('%s %.3f %.1f %.1f %.1f %.1f\n' % (filename, float(columns[5]), x1, y1, x2, y2))

        # 处理预测数据和训练数据重叠部分（将这部分数据置信度设置为1）
        # train_labels_dir = '../data/train/labels/'
        # for filename2 in find_all_file(train_labels_dir):
        #     if filename == filename2:
        #         txt_path = train_labels_dir + filename + '.txt'
        #         with open(txt_path, 'r') as f:
        #             data_lines = f.readlines()
        #             for data in data_lines:
        #                 columns = data.split(' ')
        #                 x_center, y_center, width, height = tuple(list(map(float, columns[1:])))
        #                 # print(x_center, y_center, width, height)
        #                 x1, y1, x2, y2 = re_normalized(img_width, img_height, x_center, y_center, width, height)
        #                 result_txt_lines.append('%s 1.000 %.1f %.1f %.1f %.1f\n' % (filename, x1, y1, x2, y2))
        #         break
    with open(result_txt, 'x') as f:
        for line in result_txt_lines:
            print(line)
            f.write(line)


if __name__ == '__main__':
    yolo_to_point()
