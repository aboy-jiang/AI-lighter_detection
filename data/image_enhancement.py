import os
import base64
from aip import AipImageProcess
import cv2 as cv
import shutil

# 获取目录下所有文件名
def find_all_file(dir):
    for root, ds, fs in os.walk(dir):
        for f in fs:
            filename = f.split('.')[0]
            yield filename


# 读取图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 对比度增强
def contrastEnhance(img_path, img_enhancement_path):
    APP_ID = '23412451'
    API_KEY = 'lWe6tLuOlH9cwXT0eMG7W2RE'
    SECRET_KEY = 'hckl7PHNTxfDukkZZv4eIQuYWAeyS454'
    client = AipImageProcess(APP_ID, API_KEY, SECRET_KEY)

    image = get_file_content(img_path)
    result = client.contrastEnhance(image)  # 调用图像对比度增强
    base64str = result["image"]
    image_data = base64.b64decode(base64str)
    with open(img_enhancement_path, 'wb') as f:
        f.write(image_data)


# 图片翻转：水平翻转（h）、垂直翻转（v）、水平垂直翻转（hv）
def flip(image_path, label_path):
    flips = ['h', 'v', 'hv']
    flips_code = [1, 0, -1] # flip code: h=1, v=0, hv=-1
    img = cv.imread(image_path)
    img_h, img_w, img_c = img.shape
    for i in range(3):
        # 翻转图片、保存
        flip_operation = flips[i]
        flip_code = flips_code[i]
        img_flip = cv.flip(img, flip_code)

        flip_img_path = image_path.replace('.jpg', f'_{flip_operation}.jpg')
        # 保存图片
        cv.imwrite(flip_img_path, img_flip)

        # 计算坐标、保存
        label_lines = []
        with open(label_path, 'r') as f:
            data_lines = f.readlines()
            for data in data_lines:
                columns = data.split(' ')
                x_center, y_center, width, height = tuple(list(map(float, columns[1:])))
                if 'h' in flip_operation:
                    x_center = 1 - x_center
                if 'v' in flip_operation:
                    y_center = 1 - y_center
                label_lines.append('0 %s %s %s %s\n' % (x_center, y_center, width, height))

        label_save_paht = label_path.replace('.txt', f'_{flip_operation}.txt')
        with open(label_save_paht, 'w') as f:
            for line in label_lines:
                f.write(line)


def train_data_enhancement():
    images_dir = './train/images/'
    images_enhancement_dir = './train/images_enhancement/'
    labels_dir = './train/labels/'

    for filename in find_all_file(images_dir):
        img_path = images_dir + filename + '.jpg'
        img_enhancement_path = images_enhancement_dir + filename + '.jpg'
        label_path = labels_dir + filename + '.txt'

        # 1. 对比度增强
        # contrastEnhance(img_path, img_enhancement_path)

        # 2. 图片翻转
        # flip(img_enhancement_path, label_path)


def psedudo_label_enhancement():
    detect_images_dir = './detect/'
    exp_labels_dir = './pseudo_label/exp/labels/'

    images_dir = './pseudo_label/images/'
    labels_dir = './pseudo_label/labels/'


    for filename in find_all_file(exp_labels_dir):
        detect_images_path = detect_images_dir + filename + '.jpg'
        exp_labels_path = exp_labels_dir + filename + '.txt'

        img_path = images_dir + filename + '_test.jpg'
        label_path = labels_dir + filename + '_test.txt'

        # 1. 格式化Label：计算坐标、保存
        label_lines = []
        with open(exp_labels_path, 'r') as f:
            data_lines = f.readlines()
            for data in data_lines:
                columns = data.split(' ')
                x_center, y_center, width, height = tuple(list(map(float, columns[1:5])))
                label_lines.append('0 %s %s %s %s\n' % (x_center, y_center, width, height))

        with open(label_path, 'w') as f:
            for line in label_lines:
                f.write(line)

        # 2. 移动图片
        shutil.copyfile(detect_images_path, img_path)

        # 3. 图片翻转
        flip(img_path, label_path)


if __name__ == '__main__':
    psedudo_label_enhancement()

    # 测试数据中，有一部分来自训练集
    # detect_images_dir = './detect/'
    # train_images_dir = './train/images/'
    # count = 0
    # for filename in find_all_file(detect_images_dir):
    #     for filename2 in find_all_file(train_images_dir):
    #         if (filename == filename2):
    #             count += 1
    #             # print(filename)
    # print(count)