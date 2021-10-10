import caffe
import numpy as np

# 人脸识别
def load_caffe_model(prototxt_path, caffemodel_path):
    model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)  # 加载model和network
    return model

def caffe_inference(model, img_arr):
    model.blobs['data'].data[...] = img_arr  # 执行上面设置的图片预处理操作，并将图片载入到blob中
    result = model.forward() # 输出四个分支
    y_bboxes = result['loc_branch_concat']
    y_scores = result['cls_branch_concat']
    return y_bboxes, y_scores