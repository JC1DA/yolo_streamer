#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import multiprocessing
from multiprocessing import Process, Manager, Lock, Value

from processes.process_batcher import process_batcher
from processes.process_detector import process_detector
from processes.process_downloader import process_downloader
from processes.process_postprocessor import process_postprocessor
from processes.process_resizer import process_resizer
from processes.process_visualizer import process_visualizer
from processes.process_streamer import process_streamer

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=1EiC9bvVGnk')
    parser.add_argument('--pb_file', type=str, default='./models/yolov4_320_norm.pb')
    parser.add_argument('--music_path', type=str, default='./music/music.mp3', help='path to background music')
    parser.add_argument('--output_path', type=str, default='./output_test.mp4', help='can be livestream url or file path')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--model_input_size', type=int, default=320, help='')
    parser.add_argument('--expected_fps', type=float, default=30, help='')
    parser.add_argument('--frame_height', type=int, default=720, help='')
    parser.add_argument('--frame_width', type=int, default=1280, help='')
    parser.add_argument('--auto_restart', default=False, action='store_true')
    args, unknown = parser.parse_known_args()

    #setting log
    multiprocessing.log_to_stderr()
    
    with Manager() as manager:
        stop_process = Value('i', False)
        auto_restart = Value('i', args.auto_restart)

        queues = [manager.list() for i in range(6)]

        locks = [Lock() for i in range(6)]

        #create processes
        p1 = Process(target=process_downloader, args=(stop_process, auto_restart, args.url, args.expected_fps, queues[0], locks[0], '720p'))
        p2 = Process(target=process_resizer, args=(stop_process, args.model_input_size, queues[0], locks[0], queues[1], locks[1]))
        p3 = Process(target=process_batcher, args=(stop_process, args.batch_size, queues[1], locks[1], queues[2], locks[2]))
        p4 = Process(target=process_detector, args=(stop_process, args.pb_file, queues[2], locks[2], queues[3], locks[3]))
        p5 = Process(target=process_postprocessor, args=(stop_process, args.model_input_size, queues[3], locks[3], queues[4], locks[4]))
        p6 = Process(target=process_visualizer, args=(stop_process, queues[4], locks[4], queues[5], locks[5]))
        p7 = Process(target=process_streamer, args=(stop_process, auto_restart, queues[5], locks[5], args.output_path, args.music_path, args.expected_fps, args.frame_width, args.frame_height))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()

if __name__ == "__main__":
    main()