# -*- encoding=utf-8
import numpy as np

"""
NMS包含很多框，其坐标为（x1,y1,x2,y2）,每个框对应了一个score，
我们将按照score得分降序，并将第一个最高的score的框（我们叫做标准框）作为标准框与其它框对比，
即计算出其它框与标准框的IOU值，然后设定阈值，与保留框的最大数量，若超过阈值，就删除该框，
以此类推，所选框最大不能超出设定的数量，最后得到保留的框，结束NMS
"""


def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    # topk可以将高维数组沿某一维度（该维度共N项），选出最大（最小）的K项并排序。返回排序结果和index信息。
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    # 如果没有box，返回空list
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    # 取置信度最大的（即第一个）框
    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []  # 需要保留的bounding box
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    # 按照置信度大小排序
    idxs = np.argsort(confidences)

    # 分配最后一个（得分最低）index给i，并使用pick收集这个index(即i)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        # 在得分大于当前i的boxes中，找到重合部分的左上点和右下点
        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        # 计算的得到的重合面积
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        # 计算重合度
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        # 删除得分最高的项（循环开始已经收集了）
        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        # 删除
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]
