import cv2
import time
import numpy as np

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious

def non_max_suppression(boxes, scores, threshold):	
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)

def process_single_output(boxes, scores, model_input_size, score_threshold=0.5, iou_threshold=0.5, num_classes=80):
    mask = scores >= score_threshold

    boxes_ = []
    scores_ = []
    classes_ = []

    #normalize boxes to [0-1] range
    boxes /= float(model_input_size) + 0.5

    for c in range(num_classes):
        class_boxes = boxes[mask[:,c]]
        class_box_scores = scores[:, c][mask[:,c]]

        nms_index = non_max_suppression(class_boxes, class_box_scores, iou_threshold)        

        if len(nms_index) > 0:
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = np.ones_like(class_box_scores, dtype=np.int32) * c
            boxes_.extend(class_boxes)
            scores_.extend(class_box_scores)
            classes_.extend(classes)

    return boxes_, scores_, classes_

def process_postprocessor(stop_process, model_input_size, input_queue, input_queue_lock, output_queue, output_queue_lock):

    lat_hist = []
    last_report_time = time.time()

    while not stop_process.value:
        t0 = time.time()

        batch_frames = None
        batch_scores = None
        batch_boxes  = None

        #read data from input_queue
        input_queue_lock.acquire()
        if len(input_queue) > 0:
            batch_frames, batch_scores, batch_boxes = input_queue.pop(0)
        input_queue_lock.release()

        if isinstance(batch_frames, type(None)):
            time.sleep(0.01)
            continue

        for idx in range(len(batch_frames)):
            boxes, scores, classes = process_single_output(batch_boxes[idx], batch_scores[idx], model_input_size)

            names = [class_names[class_id] for class_id in classes]

            output_queue_lock.acquire()
            output_queue.append((batch_frames[idx], scores, boxes, names))
            output_queue_lock.release()

        lat = time.time() - t0
        lat_hist.append(lat)
        if len(lat_hist) > 10:
            lat_hist.pop(0)

        if time.time() - last_report_time > 5:
            fps = len(batch_frames) / np.mean(lat_hist)
            print('process_postprocessor: fps={:.3f}'.format(fps))
            last_report_time = time.time()

        