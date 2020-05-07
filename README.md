# Introduction 
Live Streaming Yolov4-Based Object Detector on Youtube

# Requirements
Good GPU: Nvidia 1080Ti or better

Strong network connection

# Download Models and Music
```
bash download.sh
```

# Required Dependencies
```
ffmpeg with libx264 for youtube live streaming
```

# Install Python Dependencies
```
pip3 install -r requirements.txt
```

# Run Example

This example downloads 720p frames from Youtube, does detection and live stream the detections on Youtube.

```
python main.py --auto_restart --url https://www.youtube.com/watch?v=1EiC9bvVGnk --pb ./models/yolov4_320_norm.pb --model_input_size 320 --batch_size 16 --expected_fps 30 --output_path rtmp://a.rtmp.youtube.com/live2/<YOUTUBE-KEYS>
```

Or you can save the detections to a video file
```
python main.py --auto_restart --url https://www.youtube.com/watch?v=1EiC9bvVGnk --pb ./models/yolov4_320_norm.pb --model_input_size 320 --batch_size 16 --expected_fps 30 --output_path ./video.mp4
```

# Models With Larger Input Size

```
https://meowtek.art/jc1da
```

# References
Model: https://github.com/Ma-Dan/keras-yolo4

Paper: https://arxiv.org/abs/2004.10934