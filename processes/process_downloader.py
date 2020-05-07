
import streamlink
import cv2
import time

def process_downloader(stop_process, auto_restart, youtube_url, expected_fps, output_queue, output_queue_lock, quality='720p'):
    while True:
        streams = streamlink.streams(youtube_url)
        stream_url = streams[quality].url    
        cap = cv2.VideoCapture(stream_url)

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))

        drop_frame_step = 0 if expected_fps >= video_fps else video_fps / (video_fps - expected_fps)

        count = 0
        while not stop_process.value:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            count = count % video_fps

            if drop_frame_step > 0 and count % drop_frame_step == 0:
                #drop this frame to make sure we have expected fps
                continue

            output_queue_lock.acquire()
            output_queue.append(frame)
            output_queue_lock.release()            

        if auto_restart.value:
            #sleep for a while before re-connecting to the stream
            time.sleep(60)
        else:
            break

    stop_process.value = True
    print('Frame Downloader is exitting!!!')
