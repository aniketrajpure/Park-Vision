import cv2
import torch
import numpy as np
from flask import Flask, jsonify
from threading import Thread, Lock
from flask_cors import CORS 
import os
import shutil

app = Flask(__name__)
CORS(app)

class YOLOv5Model:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

    def detect_objects(self, frame):
        return self.model(frame)

class ParkingLot:
    def __init__(self, video_path, coordinates_path):
        self.video_path = video_path
        self.coordinates_path = coordinates_path
        self.cap = cv2.VideoCapture(video_path)
        self.parking_areas = self.load_parking_areas()

    def load_parking_areas(self):
        parking_areas = {}
        with open(self.coordinates_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                if len(data) < 2: 
                    continue
                area_id = data[0]
                points = [(int(data[i]), int(data[i+1])) for i in range(1, len(data), 2)]
                parking_areas[area_id] = points
        return parking_areas

    def release(self):
        self.cap.release()

class ObjectDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, frame):
        return self.model.detect_objects(frame)

class ParkingStatusUpdater:
    def __init__(self, parking_lot, object_detector):
        self.parking_lot = parking_lot
        self.object_detector = object_detector
        self.parking_status = {area_id: False for area_id in self.parking_lot.parking_areas.keys()}
        self.status_lock = Lock()

    def update_status(self):
        while self.parking_lot.cap.isOpened():
            ret, frame = self.parking_lot.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1080, 720))  
            results = self.object_detector.detect(frame)
            self.update_parking_status(results)

        self.parking_lot.release()
        cv2.destroyAllWindows()

    def update_parking_status(self, results):
        parking_status = {}
        for area, coords in self.parking_lot.parking_areas.items():
            car_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                if self.object_detector.model.names[int(cls)] == 'car':
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)

                    if cv2.pointPolygonTest(np.array(coords), (cx, cy), False) == 1:
                        xmin, ymin, xmax, ymax = [int(i) for i in xyxy]
                        label = f'{self.object_detector.model.names[int(cls)]} {conf:.2f}'
                        #reffernce here

                        car_area = (xmax - xmin) * (ymax - ymin)
                        parking_area = cv2.contourArea(np.array(coords))
                        overlap_ratio = car_area / parking_area
                        if overlap_ratio >= 0.5:
                            car_detected = True
                            break

            parking_status[area] = not car_detected

        with self.status_lock:
            self.parking_status = parking_status

    def get_parking_status(self):
        with self.status_lock:
            return self.parking_status

@app.route('/parking_status', methods=['GET'])
def get_parking_status():
    return jsonify(parking_status_updater.get_parking_status())

if __name__ == '__main__':
    video_path = 'frontcar.mp4'
    coordinates_path = 'polygon_coordinates.txt'

    parking_lot = ParkingLot(video_path, coordinates_path)
    model = YOLOv5Model()
    object_detector = ObjectDetector(model)
    parking_status_updater = ParkingStatusUpdater(parking_lot, object_detector)

    update_thread = Thread(target=parking_status_updater.update_status)
    update_thread.start()

    app.run(debug=True)


