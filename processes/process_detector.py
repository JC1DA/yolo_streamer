import cv2
import time
import numpy as np
import tensorflow as tf

def process_detector(stop_process, pb_path, input_queue, input_queue_lock, output_queue, output_queue_lock):

    tf_graph = tf.Graph()
    tf_config = tf.ConfigProto()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    tf_config.gpu_options.allow_growth = True
   
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf_graph.as_default() as graph:
        tf.import_graph_def(graph_def, name='')    

    #tf_sess = tf.Session(graph=tf_graph,config=tf.ConfigProto(gpu_options=gpu_options))
    tf_sess = tf.Session(graph=tf_graph,config=tf_config)

    input_tensor = tf_graph.get_tensor_by_name('inputs:0')
    boxes_tensor = tf_graph.get_tensor_by_name('boxes:0')
    scores_tensor = tf_graph.get_tensor_by_name('scores:0')

    lat_hist = []
    last_report_time = time.time()
    
    while not stop_process.value:
        t0 = time.time()

        frames = None
        resized_frames = None
        #read data from input_queue
        input_queue_lock.acquire()
        if len(input_queue) > 0:
            if len(input_queue) > 4:
                #drop some batch
                for i in range(len(input_queue) // 2):
                    input_queue.pop(0)
            frames, resized_frames = input_queue.pop(0)
        input_queue_lock.release()

        if isinstance(frames, type(None)):
            time.sleep(0.01)
            continue

        out_scores, out_boxes = tf_sess.run([scores_tensor, boxes_tensor], feed_dict={input_tensor: resized_frames})

        #forward data to next process
        output_queue_lock.acquire()
        output_queue.append((frames, out_scores, out_boxes))
        output_queue_lock.release()

        lat = time.time() - t0
        lat_hist.append(lat)
        if len(lat_hist) > 10:
            lat_hist.pop(0)

        if time.time() - last_report_time > 5:
            fps = len(frames) / np.mean(lat_hist)
            print('process_detector: fps={:.3f}'.format(fps))
            last_report_time = time.time()



    