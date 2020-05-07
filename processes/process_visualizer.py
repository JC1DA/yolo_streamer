import cv2
import time
import os
import numpy as np

def process_visualizer(stop_process, input_queue, input_queue_lock, output_queue, output_queue_lock):
    count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.75

    while not stop_process.value:
        frame = None
        scores = None
        boxes  = None
        names  = None

        #read data from input_queue
        input_queue_lock.acquire()
        if len(input_queue) > 0:
            frame, scores, boxes, names = input_queue.pop(0)
        input_queue_lock.release()

        if isinstance(frame, type(None)):
            time.sleep(0.01)
            continue

        h,w = frame.shape[:2]

        for box, score, class_name in zip(boxes, scores, names):    
            top, left, bottom, right = box

            left   = int(left * w)
            right  = int(right * w)
            top    = int(top * h)
            bottom = int(bottom * h)

            frame = cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 1)
            frame = cv2.putText(frame, class_name, (left, top), font, fontScale, (0,255,0), 1, cv2.LINE_AA)

        #cv2.imwrite(os.path.join(tmp_dir, str(count).zfill(9) + '.jpg'), frame)
        
        output_queue_lock.acquire()
        output_queue.append(frame)
        output_queue_lock.release()

        count += 1
            