import time
import os
import subprocess as sp
import numpy as np

from subprocess import DEVNULL

class sleeper():
    def __init__(self, name, expected_fps, min_sleep_time=0.01, max_sleep_time=0.03, buffer_size=1000, debug=False):
        self.name = name
        self.expected_fps = expected_fps
        self.min_sleep_time = min_sleep_time
        self.max_sleep_time = max_sleep_time
        self.buffer_size = buffer_size
        self.buffer = []
        self.sleep_time = 1.0 / expected_fps
        self.debug_counter = 0
        self.debug = debug

    def update(self, processing_lat):
        self.buffer.append(processing_lat)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        #update sleeptime
        avg_lat = np.mean(self.buffer)
        self.sleep_time = (1.0 / self.expected_fps) - avg_lat
        self.sleep_time = min(self.sleep_time, self.max_sleep_time)

        self.debug_counter += 1

        if self.debug_counter > 100:
            #reset debug_counter
            self.debug_counter = 0
            # if self.debug:
            #     print('Update {}-sleeptime to {:.2f} ms\n'.format(self.name, self.sleep_time * 1000))

    def sleep(self):
        if self.sleep_time > self.min_sleep_time:
            time.sleep(self.sleep_time)

def build_ffmpeg_cmd(output_path, music_path, expected_fps, width, height):
    dimension = '{}x{}'.format(width, height)
    command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', dimension,
            '-pix_fmt', 'bgr24',
            '-i', '-',
            '-stream_loop', '-1',
            '-i', music_path,
            '-shortest',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:a', 'aac',
            '-threads', '4',
            '-r', str(expected_fps),
            '-g', str(expected_fps),
            '-f', 'flv',
            output_path
    ]

    return command

def process_streamer(stop_process, auto_restart, input_queue, input_queue_lock, \
                        output_path, music_path, expected_fps, width, height):
    
    sleeper_ = sleeper('Streamer', expected_fps=expected_fps*1.1, min_sleep_time=0.01, max_sleep_time=0.03, buffer_size=1000, debug=True)
    
    while True:
        ffmpeg_command = build_ffmpeg_cmd(output_path, music_path, expected_fps, width, height)
        proc = sp.Popen(ffmpeg_command, stdin=sp.PIPE, stderr=DEVNULL, stdout=DEVNULL)
        #proc = sp.Popen(ffmpeg_command, stdin=sp.PIPE)

        while not stop_process.value:
            t0 = time.time()

            frame = None

            input_queue_lock.acquire()
            if len(input_queue) > 0:
                frame = input_queue.pop(0)
            input_queue_lock.release()

            if not isinstance(frame, type(None)):
                try:
                    os.write(proc.stdin.fileno(), frame.tostring())
                except:
                    #uploading error, restart the stream
                    break

            sleeper_.update(time.time() - t0)
            sleeper_.sleep()

        try:
            proc.stdin.close()
            proc.wait()
        except:
            pass

        if auto_restart.value:
            #dump all current frames before re-starting
            input_queue_lock.acquire()
            while len(input_queue) > 0:
                input_queue.pop()
            input_queue_lock.release()
        else:
            break

    stop_process.value = True
    print('Process streamer is exiting!!!')