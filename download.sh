#!/bin/bash

mkdir -p models
wget http://meowtek.art/jc1da/yolov4_320_norm.pb --no-check-certificate -O models/yolov4_320_norm.pb

mkdir -p music
wget https://meowtek.art/jc1da/music.mp3 --no-check-certificate -O music/music.mp3
