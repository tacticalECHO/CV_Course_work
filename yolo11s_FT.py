import cv2
import numpy as np
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

"""
YOLO models are downloaded from official YOLO repository.
github: https://github.com/ultralytics/ultralytics?tab=readme-ov-file
"""
def bbox_iou(box1, box2):
    """calculates the IoU of two bounding boxes"""
    # box1 and box2 are in xywh format (x, y, w, h)
    # change to xyxy format (x1, y1, x2, y2)
    box1 = np.array([box1[0]-box1[2]/2, box1[1]-box1[3]/2,
                    box1[0]+box1[2]/2, box1[1]+box1[3]/2])
    box2 = np.array([box2[0]-box2[2]/2, box2[1]-box2[3]/2,
                    box2[0]+box2[2]/2, box2[1]+box2[3]/2])
    
    # calculate the intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

class KalmanBoxTracker:
    """This class represents the Kalman filter for tracking a single object."""
    count = 0  # ID counter
    
    def __init__(self, init_box):
        # initialize the ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        # initialize the Kalman filter
        self.KalmanBoxTracker = KalmanFilter(dim_x=6, dim_z=4)
        
        # This is the state transition matrix
        # This matrix is used to predict the next state based on the current state
        # The state vector is [x, y, w, h, vx, vy]
        self.KalmanBoxTracker.F = np.array([
            [1,0,0,0,1,0],
            [0,1,0,0,0,1],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])
        
        # This is the observation matrix
        # This matrix is used to map the state vector to the observed measurements
        # The observation vector is [x, y, w, h]
        self.KalmanBoxTracker.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0]
        ])

        # This is the covariance matrix
        # It represents the uncertainty in the state vector
        # The diagonal elements represent the uncertainty in each state variable
        self.KalmanBoxTracker.P[4:,4:] *= 1000  # greater uncertainty in velocity
        self.KalmanBoxTracker.P *= 10
        
        # This is the process noise covariance matrix
        # It represents the uncertainty in the process model
        # The diagonal elements represent the uncertainty in each state variable
        self.KalmanBoxTracker.Q = np.eye(6)
        self.KalmanBoxTracker.Q[4:,4:] *= 0.01
        
        # This is the measurement noise covariance matrix
        # It represents the uncertainty in the measurements
        self.KalmanBoxTracker.R = np.diag([5, 5, 3, 3]) 
        
        # Initialize the state vector with the initial bounding box
        # The state vector is [x, y, w, h, vx, vy]
        self.KalmanBoxTracker.x = self.convert_bbox_to_z(init_box)
        
        # Initialize the ID and consecutive misses
        self.consecutive_misses = 0

    @staticmethod
    def convert_bbox_to_z(bbox):
        # Convert the bounding box to the state vector
        return np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0]])
    @staticmethod
    def convert_x_to_bbox(x):
        # Convert the state vector to the bounding box
        return np.array([x[0][0], x[1][0], x[2][0], x[3][0]])
    def predict(self):
        #"""Predict the next state using the Kalman filter"""
        self.kf.predict()
        return self.convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        # update the Kalman filter with the new bounding box
        z = self.convert_bbox_to_z(bbox)[:4]
        self.kf.update(z)
        self.consecutive_misses = 0 
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    # This function associates detections with trackers using the Hungarian algorithm.
    # It returns a list of matched indices.
    # change detections and trackers to numpy arrays
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int)
    
    # calculate the IoU matrix
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = bbox_iou(det, trk)
    
    # Use the Hungarian algorithm to find the best matches
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))
    
    # filter out matches below the IoU threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            continue
        matches.append(m.reshape(1,2))
    
    return np.concatenate(matches, axis=0) if matches else np.empty((0,2), dtype=int)

# Initialize YOLO model and video capture
model = YOLO("best.pt")
video_path = "Own_test_4.mp4"
output_path = "Own_test_4_output.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) # the output video settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize trackers and history
trackers = dict()
max_history = 30
track_history = defaultdict(list)
color_map = defaultdict(lambda: tuple(np.random.randint(0,255,3).tolist()))

detect_interval = 1  # every 1 frame to detect
frame_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # predict
    trk_ids = []  # store the IDs of the trackers
    trk_boxes = []
    for trk_id in trackers.keys():
        pred_box = trackers[trk_id].predict()
        trk_boxes.append(pred_box)
        trk_ids.append(trk_id)
    det_boxes = []# detect boxes
    if frame_counter % detect_interval == 0:
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.5,
            iou=0.3,
            verbose=False
        )
        if results[0].boxes.id is not None:
            det_boxes = results[0].boxes.xywh.cpu().numpy()
    
    # get the matched results
    matched = associate_detections_to_trackers(det_boxes, trk_boxes)

    for m in matched: # update the matched trackers
        det_idx, trk_idx = m[0], m[1]
        trk_id = list(trackers.keys())[trk_idx]
        trackers[trk_id].update(det_boxes[det_idx])
    
    # if there are unmatched detections, create new trackers
    unmatched_detections = set(range(len(det_boxes))) - set(matched[:,0])
    for d_idx in unmatched_detections:
        new_trk = KalmanBoxTracker(det_boxes[d_idx])
        trackers[new_trk.id] = new_trk
    
    # process unmatched trackers
    unmatched_trackers = set(range(len(trk_boxes))) - set(matched[:,1])
    for t_idx in unmatched_trackers:
        trk_id = trk_ids[t_idx]  # get the ID of the unmatched tracker
        trackers[trk_id].consecutive_misses += 1
        if trackers[trk_id].consecutive_misses > 20:
            del trackers[trk_id]
            del track_history[trk_id]
    
    # Visualize the results
    annotated_frame = frame.copy()
    for trk_id, tracker in trackers.items():
        x, y, w, h = tracker.kf.x[:4].flatten()
        cv2.rectangle(annotated_frame,(int(x-w/2), int(y-h/2)),(int(x+w/2), int(y+h/2)),color_map[trk_id], 2)# draw the box
        vx, vy = tracker.kf.x[4:6].flatten()
        cv2.arrowedLine(annotated_frame,(int(x), int(y)),(int(x+vx*10), int(y+vy*10)),(0,255,0), 2)# draw the velocity vector
        center = (float(x), float(y))
        track_history[trk_id].append(center)
        if len(track_history[trk_id]) > max_history:
            track_history[trk_id].pop(0)
        points = np.array(track_history[trk_id], dtype=np.int32)
        cv2.polylines(annotated_frame, [points], False, color_map[trk_id], 3)
        speed = np.sqrt(vx**2 + vy**2)
        cv2.putText(annotated_frame,f"ID:{trk_id}",(int(x-w/2), int(y-h/2)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,255,255), 2)# draw the ID
    
    out.write(annotated_frame)# write the frame to the output video
    cv2.imshow("Optimized Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_counter += 1

cap.release()
out.release()
cv2.destroyAllWindows()