from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tempfile

from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)
# Load YOLO model
net = cv2.dnn.readNet("app/yolov3.weights", "app/yolov3.cfg")
classes = []
with open("app/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define vehicle classes based on COCO dataset (e.g., cars, trucks, buses)
vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

def count_vehicles_from_video(video_path):
    video = cv2.VideoCapture(video_path)
    total_vehicle_count = 0
    frame_count = 0
    skip = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if skip % 10 == 0:
            # Count vehicles in the current frame
            vehicle_count = count_vehicles(frame)
            total_vehicle_count += vehicle_count
            frame_count += 1
        skip += 1

    video.release()
    
    average_vehicle_count = total_vehicle_count / frame_count if frame_count > 0 else 0
    return total_vehicle_count, average_vehicle_count

def count_vehicles(image):
    height, width, channels = image.shape

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    vehicle_count = 0
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out vehicles based on confidence threshold
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                vehicle_count += 1

    return vehicle_count

@app.route('/count_vehicles', methods=['POST'])
def detect_vehicles_from_videos():
    import requests
    try:
        # Get the video files from the request
        files = request.files
        
        directions = ['north', 'south', 'east', 'west']
        vehicle_counts = {}
        average_vehicle_counts = {}

        for direction in directions:
            if direction in files:
                # Save the video to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(files[direction].read())
                temp_file.close()
                
                # Count vehicles from the video
                total_vehicle_count, avg_vehicle_count = count_vehicles_from_video(temp_file.name)
                
                # Store the results
                vehicle_counts[direction] = total_vehicle_count
                average_vehicle_counts[direction] = avg_vehicle_count

                # Delete the temporary file
                os.unlink(temp_file.name)
            else:
                vehicle_counts[direction] = 0
                average_vehicle_counts[direction] = 0
        
        # Prepare data to send to the second microservice
        vehicle_data = {
            'vehicles': average_vehicle_counts
        }

        # Send the vehicle data to the second microservice
        response = requests.post('http://adaptive-time-ms:5001/green-time', json=vehicle_data)
        adaptive_times = response.json()

        return jsonify({
            'vehicle_counts': vehicle_counts,
            'average_vehicle_counts_per_frame': average_vehicle_counts,
            'adaptive_green_times': adaptive_times,
            'status': 'success'
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e), 'status': 'failure'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
