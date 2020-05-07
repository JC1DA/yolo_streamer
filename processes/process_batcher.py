import cv2
import time

ORIGINAL_FRAMES = []
RESIZED_FRAMES  = []

def process_batcher(stop_process, batch_size, input_queue, input_queue_lock, output_queue, output_queue_lock):
    global ORIGINAL_FRAMES
    global RESIZED_FRAMES

    while not stop_process.value:
        frame = None
        resized_frame = None
        #read data from input_queue
        input_queue_lock.acquire()
        if len(input_queue) > 0:
            frame, resized_frame = input_queue.pop(0)
        input_queue_lock.release()

        if isinstance(frame, type(None)):
            time.sleep(0.01)
            continue

        ORIGINAL_FRAMES.append(frame)
        RESIZED_FRAMES.append(resized_frame)

        if len(ORIGINAL_FRAMES) == batch_size:
            #forward to next process
            output_queue_lock.acquire()
            orginal_batch = ORIGINAL_FRAMES
            resized_batch = RESIZED_FRAMES
            output_queue.append((orginal_batch, resized_batch))
            output_queue_lock.release()

            ORIGINAL_FRAMES = []
            RESIZED_FRAMES  = []