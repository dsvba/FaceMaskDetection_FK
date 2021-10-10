# -*- coding:utf-8 -*-
import cv2
import time
import argparse

import numpy as np
from PIL import Image
# from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# 用于推断，批大小为1，模型输出形状为[1，N，4]，因此将锚点的dim扩展为[1，anchor_num，4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    检测推理的主要功能
    :param image: 3D numpy 图片数组
    :param conf_thresh: 分类概率大最小阈值
    :param iou_thresh: 网管的IOU门限
    :param target_shape: 输入模型的大小
    :param draw_result: 是否将边界框拖入图像
    :param show_result: 是否显示图像
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    # opencv库当中的resize函数可以帮助我们快速进行图像的大小变化
    # 默认的插值方法为：双线性插值
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)  # 在0位置添加数据
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # 删除批次尺寸，因为批次始终为1进行推断。
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # 为了加快速度，请执行单类NMS，而不是多类NMS。
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx是nms之后的活动边界框
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # 裁剪坐标，避免该值超出图像边界
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        # 用Opencv画矩形框
        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info


# Opencv获取视频信息
def run_on_video(video_path, output_video_name, conf_thresh):
    # 创建VideoCapture，传入0即打开系统默认摄像头
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        # 就是处理一帧的例子，这里转为灰度图
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            # 这个函数之后应接cv2.waitKey函数来显示指定图像。否则，它不会显示图像
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()


if __name__ == "__main__":
    # argparse模块可以让人轻松编写用户友好的命令行接口
    # 首先创建解析器对象
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    # 添加参数
    parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    # 解析参数
    args = parser.parse_args()
    # 方便使用定义好的超参数
    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, show_result=True, target_shape=(260, 260))
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)
