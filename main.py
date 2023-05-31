import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
import argparse
import torch.nn as nn
import shutil
from itertools import groupby
import re
import imutils

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载预训练的ResNet101模型
resnet = models.resnet101(pretrained=True)
# 去除最后一层全连接层
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
# 将模型设置为评估模式
resnet.eval()


def resize(np_image, size=800):
    """
    调整图片大小
    :param size: target size
    :param np_image: target image (np array)
    :return: result image (np array)
    """
    height, width = np_image.shape[0:2]
    # 调整图片大小
    target_width = size
    target_height = size
    if width >= height and width != target_width:
        np_image = cv2.resize(np_image, (target_width, int(height * target_width / width)))
    elif width < height != target_height:
        np_image = cv2.resize(np_image, (int(width * target_height / height), target_height))
    return np_image


def resize1(np_image1, np_image2, size=800):
    """
    调整图片大小
    :param size: target size
    :param np_image1: target image1 (np array)
           np_image2: target image1 (np array)
    :return: result image (np array)
    """

    height1, width1 = np_image1.shape[0:2]
    height2, width2 = np_image2.shape[0:2]
    # print("height1:", height1)
    # print("width1:", width1)
    # print("height2:", height2)
    # print("width2:", width2)
    # print(" ")
    # 调整图片大小
    if height1 == height2:
        if width1 == width2:
            scale = min(size / height1, size / width1)
            target_width = int(scale * width1)
            target_height = int(scale * height1)
            np_image1 = cv2.resize(np_image1,
                                   (target_width, target_height),
                                   cv2.INTER_CUBIC)
            np_image2 = cv2.resize(np_image2,
                                   (target_width, target_height),
                                   cv2.INTER_CUBIC)
        else:
            width = min(width1, width2)
            scale = min(size / height1, size / width)
            target_width1 = int(scale * width1)
            target_width2 = int(scale * width2)
            target_height = int(scale * height1)
            np_image1 = cv2.resize(np_image1,
                                   (target_width1, target_height),
                                   cv2.INTER_CUBIC)
            np_image2 = cv2.resize(np_image2,
                                   (target_width2, target_height),
                                   cv2.INTER_CUBIC)
    else:
        height = 450
        scale1 = min(size / height, size / width1)
        scale2 = min(size / height, size / width2)
        target_width1 = int(scale1 * width1)
        target_width2 = int(scale2 * width2)
        np_image1 = cv2.resize(np_image1,
                               (target_width1, height),
                               cv2.INTER_CUBIC)
        np_image2 = cv2.resize(np_image2,
                               (target_width2, height),
                               cv2.INTER_CUBIC)
        height1, width1 = np_image1.shape[0:2]
        height2, width2 = np_image2.shape[0:2]
        # print("height1:", height1)
        # print("width1:", width1)
        # print("height2:", height2)
        # print("width2:", width2)
    return np_image1, np_image2


def sift_detect(img):
    """
    sift 算子生成
    :param img: target image (np array)
    :return: kp: key_point
             des: descriptor 描述子
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # cv2.drawKeypoints(img, kp, img)
    # cv2.imshow("test", img)
    return kp, des


def read_img(n):
    feature_list = []
    images = []
    for i in range(1, n):
        # img = cv2.imread('dataset/{0}.jpg'.format(i))
        img = Image.open('dataset/{0}.jpg'.format(i)).convert('RGB')
        if img is None:
            continue
        else:
            img_tensor = preprocess(img)
            img_tensor = img_tensor.unsqueeze(0)
            # 通过ResNet101模型获取特征向量
            features = resnet(img_tensor)
            features = features.detach().squeeze()
            tmp = features.numpy()
            print(tmp.shape)
            feature_list.append(features.numpy())
    # print(len(descriptors))
    return feature_list


def sort_files(file_list):
    """
    将file_list中的文件名按数字从小到大排序
    :param file_list: target list
    :return: sorted file_list
    """

    def extract_number(filename):
        # 使用正则表达式提取文件名中的数字
        number = int(re.search(r'\d+', filename).group())
        return number

    # 使用 sorted 函数和 key 参数对文件列表进行排序
    sorted_list = sorted(file_list, key=extract_number)
    return sorted_list


def classify(n):
    """
    将图像按场景分类，并按类别分别写入cluster文件夹
    :param n: 输入图片数量(int)
    :return: None
    """
    data = read_img(n)
    kmeans = KMeans(n_clusters=4, random_state=10).fit(data)
    grouped_list = [list(group) for key, group in groupby(kmeans.labels_)]
    # print(len(grouped_list))
    # print(kmeans.labels_)

    dst_dir = "E:/Python files/quanjinpingjie/cluster/"
    src_dir = "E:/Python files/quanjinpingjie/dataset/"
    for i in range(len(grouped_list)):
        isExists = os.path.exists(dst_dir + str(i))
        if not isExists:
            # 判断如果文件不存在,则创建
            os.makedirs(dst_dir + str(i))
            # print("%s 目录创建成功" % i)
        else:
            # print("%s 目录已经存在" % i)
            continue
            # 如果文件不存在,则继续上述操作,直到循环结束
    # for label in kmeans.labels_:
    temp = sort_files(os.listdir(src_dir))
    for i in range(len(temp)):
        label = kmeans.labels_[i]
        label_dir = os.path.join(dst_dir, str(label))
        # 将图片移动到目标文件夹中
        src_path = os.path.join(src_dir, temp[i])
        dst_path = os.path.join(label_dir, temp[i])
        shutil.copy(src_path, dst_path)


def stitch(img1, img2):
    """
    全景图像生成, 基于sift以及ransac算法在拼接处按权重拼接。
    :param img1: target image1 (np array)
    :param img2: target image2 (np array)
    :return: img3: result image1 (np array), 直接拼接
             img4: result image2 (np array), 边界处根据权重拼接
    """
    kp1, descrip1 = sift_detect(img1)
    kp2, descrip2 = sift_detect(img2)
    bf = cv2.BFMatcher()
    match = bf.knnMatch(descrip1, descrip2, k=2)
    good = []
    test = []
    for i, (m, n) in enumerate(match):
        if m.distance < 0.75 * n.distance:
            test.append([m])
            good.append(m)
    # img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, test, None, flags=2)
    # cv2.imshow("BFmatch", img4)
    ransacReprojThreshold = 4
    # ransacReprojThreshold：将点对视为内点的最大允许重投影错误阈值
    if len(good) > ransacReprojThreshold:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
        # 计算变换矩阵M,使用RANSAC方法以便于反向投影错误率达到最小
        warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))
        # 透视变换

        # 若直接拼接
        # direct = warpImg.copy()
        # direct[0:image1.shape[0], 0:img1.shape[1]] = img1

        # 若按权重拼接
        rows, cols = img1.shape[:2]
        # print("rows:", rows)
        # print("cols:", cols)
        for col in range(0, cols):
            if img1[:, col].any() and warpImg[:, col].any():
                # 开始重叠的最左端
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if img1[:, col].any() and warpImg[:, col].any():
                # 重叠的最右一列
                right = col
                break
        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not img1[row, col].any():
                    # 如果没有原图，用旋转的填充
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = img1[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(img1[row, col]
                                            * (1 - alpha)
                                            + warpImg[row, col]
                                            * alpha, 0, 255)
        warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
        # img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
        # opencv读取图片为BGR非正常颜色，需转换成RGB
        img4 = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
        # Attention: 不能用cv2.imshow显示图片，颜色不对！！！
        # cv2.imshow("result", direct)
        # plt.imshow(img4)
        # plt.show()
        return img4
    else:
        print("Matches Unavailable!")


def stitch1(img1, img2, img3):
    img = stitch(img1, img2)
    # 先拼两张
    img = split(img)
    # 去除黑边
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize1(img, img3)
    result = stitch(img, img3)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imshow("Panorama", result)
    return result


def stitch2(img1, img2, img3):
    """
    全景图像生成，使用opencv自带的Stitcher类。同时对拼接后图片进行处理以去除黑边。
    :param img1: target image1 (np array)
    :param img2: target image2 (np array)
    :return: img3: result image1 (np array), 直接拼接
             img4: result image2 (np array), 边界处根据权重拼接
    """
    # 创建拼接器
    stitcher = cv2.Stitcher_create()
    # 将图像拼接为全景图像
    (status, stitched) = stitcher.stitch([img1, img2, img3], )
    # 如果拼接成功，则显示全景图像
    if status == 0:
        cv2.imshow("Panorama", stitched)

    # 去黑边
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, (0, 0, 0))
    # 给图片四周加上10像素的黑色边框
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
    # 寻找拼接后图片内部最大的矩形区域
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    stitched = stitched[y:y + h, x:x + w]
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)


def split(img):
    """
    裁剪掉拼接后图片的黑边
    :param img: target image (np array)
    :return: result image1 (np array)
    """
    m, n = img.shape[:2]
    nn = n
    mm = m
    for i in range(0, m):
        for j in range(0, n):
            if np.sum(img[i][j]) == 0:
                # 遇到黑色则开始测试
                test_col = []
                for k in range(0, 10):
                    test_col.append(np.sum(img[i + k][j]))
                if np.sum(test_col) == 0:
                    nn = j
                    break
                else:
                    continue
            else:
                continue
        break
    img = img[:mm, :nn]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    # classify(13)
#
    # data_dir = "cluster/"
    # classes = os.listdir(data_dir)
    # for cla in classes:
    #     files = os.listdir(data_dir + cla)
    #     clu = []
    #     for f in files:
    #         img = cv2.imread(data_dir + cla + "/" + f)
    #         img = resize(img)
    #         clu.append(img)
    #     stitch2(clu[0], clu[1], clu[2])
    image1 = cv2.imread("dataset/5.jpg")
    image2 = cv2.imread("dataset/4.jpg")
    image3 = cv2.imread("dataset/6.jpg")
    image1 = resize(image1)
    image2 = resize(image2)
    image3 = resize(image3)
    stitch2(image1, image2, image3)
    # result = stitch(image1, image2)
cv2.waitKey(0)
