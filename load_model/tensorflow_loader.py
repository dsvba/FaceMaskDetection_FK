# -*- encoding=utf-8 -*-
import tensorflow as tf

if tf.__version__ > '2':
    import tensorflow.compat.v1 as tf

import numpy as np

# 保存和部署模型
PATH_TO_TENSORFLOW_MODEL = 'models/face_mask_detection.pb'


def load_tf_model(tf_model_path):
    '''
    Load the model.
    :param tf_model_path: model to tensorflow model.
    :return: session and graph
    '''
    # TensorFlow模型载入内存
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # 创建一个空的GraphDef对象，用于记录TensorFlow的计算图上的节点信息
        od_graph_def = tf.GraphDef()
        # 打开pb模型，tf.gfile.GFile('path','rb')
        with tf.gfile.GFile(tf_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.Session(graph=detection_graph)
                return sess, detection_graph


# 在这个类中，创建了一个函数，在图像上运行分类，并返回图像中分类的边界框，分数
def tf_inference(sess, detection_graph, img_arr):
    '''
    Receive an image array and run inference
    :param sess: tensorflow session.
    :param detection_graph: tensorflow graph.
    :param img_arr: 3D numpy array, RGB order.
    :return:
    '''
    image_tensor = detection_graph.get_tensor_by_name('data_1:0')
    detection_bboxes = detection_graph.get_tensor_by_name('loc_branch_concat_1/concat:0')
    detection_scores = detection_graph.get_tensor_by_name('cls_branch_concat_1/concat:0')
    # image_np_expanded = np.expand_dims(img_arr, axis=0)
    bboxes, scores = sess.run([detection_bboxes, detection_scores],
                              feed_dict={image_tensor: img_arr})

    return bboxes, scores
