#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: app.py
Desc: 蓝谷计算机视觉Demo入口
Author: gaoyang
Time: 2023/11/18
"""
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading
import numpy as np 
from collections import defaultdict

#将模型、当前帧和历史轨迹设置为全局变量
track_history = defaultdict(lambda: [])
model_2detect_track = None
model_segment = None
model_pose = None
model_track = None
now_frame = None

#摄像头rtsp流地址
cap = cv2.VideoCapture("rtsp://admin:iic309311@172.16.0.25:554/Streaming/Channels/101")

app = Flask(__name__)

def get_frame_from_video():
    """
    获取最新视频帧函数
    """
    global now_frame
    while True:
        ret, frame = cap.read()
        if ret:
            now_frame = frame

def detect_frame():
    """
    图像检测部分函数
    """
    global model_2detect_track, now_frame
    results = model_2detect_track(now_frame)
    frame_2detect_track = results[0].plot()
    return frame_2detect_track

def segment_frame():
    """
    实例分割部分函数
    """
    global model_segment, now_frame
    results = model_segment(now_frame)
    frame_segment = results[0].plot()
    return frame_segment

def pose_frame():
    """
    姿态识别部分函数
    """
    global model_pose, now_frame
    results = model_pose(now_frame)
    frame_pose = results[0].plot()
    return frame_pose

def track_frame():
    """
    轨迹追踪部分函数
    """
    global model_track, now_frame, track_history

    try:
        results = model_track.track(now_frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y))) 
            if len(track) > 30:  
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    
    except:
        annotated_frame = now_frame.copy()
    
    return annotated_frame

def init_model():
    """
    模型初始化函数
    """
    global model_2detect_track, model_segment, model_pose, model_track
    model_2detect_track = YOLO('./weights/yolov8n.pt')
    model_segment = YOLO("./weights/yolov8n-seg.pt")
    model_pose = YOLO("./weights/yolov8n-pose.pt")
    model_track = YOLO('./weights/yolov8n.pt')

def generate_detect_frames():
    """
    图像检测结果推流函数
    """
    global now_frame
    while True:
        processed_frame_1 = detect_frame()
        ret, buffer = cv2.imencode('.jpg', processed_frame_1)
        processed_frame_1 = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +  processed_frame_1 + b'\r\n')
        
def generate_pose_frames():
    """
    姿态识别结果推流函数
    """
    global now_frame
    while True:
        processed_frame_2 = pose_frame()
        ret, buffer = cv2.imencode('.jpg', processed_frame_2)
        processed_frame_2 = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +  processed_frame_2 + b'\r\n')
        
def generate_segment_frames():
    """
    图像分割结果推流函数
    """
    global now_frame
    while True:
        processed_frame_3 = segment_frame()
        ret, buffer = cv2.imencode('.jpg', processed_frame_3)
        processed_frame_3 = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +  processed_frame_3 + b'\r\n')

def generate_track_frames():
    """
    轨迹追踪结果推流函数
    """
    global now_frame
    while True:
        processed_frame_4 = track_frame()
        ret, buffer = cv2.imencode('.jpg', processed_frame_4)
        processed_frame_4 = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +  processed_frame_4 + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_track')
def video_track():
    return Response(generate_track_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detect')
def video_detect():
    return Response(generate_detect_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_pose')
def video_pose():
    return Response(generate_pose_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_segment')
def video_segment():
    return Response(generate_segment_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    thread_get_frame = threading.Thread(target=get_frame_from_video, args=())
    thread_get_frame.start()

    thread_init_model = threading.Thread(target=init_model, args=())
    thread_init_model.start()

    thread_app_run = threading.Thread(target=app.run, args=('0.0.0.0', 5000, False))
    thread_app_run.start()