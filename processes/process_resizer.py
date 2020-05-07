import cv2
import time
import numpy as np

def process_resizer(stop_process, target_size, input_queue, input_queue_lock, output_queue, output_queue_lock):

    lat_hist = []
    last_report_time = time.time()

    while not stop_process.value:
        t0 = time.time()

        frame = None
        #read data from input_queue
        input_queue_lock.acquire()
        if len(input_queue) > 0:
            frame = input_queue.pop(0)
        input_queue_lock.release()

        if isinstance(frame, type(None)):
            time.sleep(0.01)
            continue

        resized_frame = cv2.resize(frame, (target_size, target_size))

        #forward two frames to next process
        output_queue_lock.acquire()
        output_queue.append((frame, resized_frame))
        output_queue_lock.release()

        lat = time.time() - t0
        lat_hist.append(lat)
        if len(lat_hist) > 10:
            lat_hist.pop(0)

        if time.time() - last_report_time > 5:
            fps = 1.0 / np.mean(lat_hist)
            print('process_resizer: fps={:.3f}'.format(fps))
            last_report_time = time.time()
