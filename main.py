
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
import threading
import time
import math
from collections import deque, defaultdict

# Initialize database
def init_db():
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            object_class TEXT,
            confidence REAL,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_w INTEGER,
            bbox_h INTEGER,
            track_id INTEGER,
            is_locked BOOLEAN DEFAULT 0,
            track_duration REAL DEFAULT 0,
            velocity_x REAL DEFAULT 0,
            velocity_y REAL DEFAULT 0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_tracks (
            track_id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_class TEXT,
            first_seen DATETIME,
            last_seen DATETIME,
            total_detections INTEGER DEFAULT 1,
            avg_confidence REAL,
            track_status TEXT DEFAULT 'active',
            is_locked BOOLEAN DEFAULT 0,
            lock_reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

class EnhancedObjectTracker:
    def __init__(self, max_disappeared=45, max_distance=120):
        self.next_track_id = 1
        self.tracked_objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.locked_objects = set()
        self.track_history = defaultdict(lambda: deque(maxlen=50))
        self.velocities = {}
        self.confidence_history = defaultdict(lambda: deque(maxlen=10))
        self.size_history = defaultdict(lambda: deque(maxlen=10))
        
        # Enhanced tracking parameters
        self.velocity_smoothing = 0.7
        self.position_smoothing = 0.3
        self.confidence_threshold = 0.3
        self.stability_threshold = 5

    def calculate_velocity(self, track_id, current_center):
        """Calculate smoothed velocity for a tracked object"""
        if track_id in self.track_history and len(self.track_history[track_id]) > 2:
            history = list(self.track_history[track_id])
            
            # Use multiple points for better velocity estimation
            if len(history) >= 3:
                recent_points = history[-3:]
                velocities = []
                
                for i in range(1, len(recent_points)):
                    prev_center = recent_points[i-1]['center']
                    curr_center = recent_points[i]['center']
                    prev_time = recent_points[i-1]['timestamp']
                    curr_time = recent_points[i]['timestamp']
                    
                    dt = curr_time - prev_time
                    if dt > 0:
                        vx = (curr_center[0] - prev_center[0]) / dt
                        vy = (curr_center[1] - prev_center[1]) / dt
                        velocities.append((vx, vy))
                
                if velocities:
                    # Average the velocities for smoother result
                    avg_vx = sum(v[0] for v in velocities) / len(velocities)
                    avg_vy = sum(v[1] for v in velocities) / len(velocities)
                    
                    # Apply smoothing
                    if track_id in self.velocities:
                        prev_vx, prev_vy = self.velocities[track_id]
                        avg_vx = self.velocity_smoothing * prev_vx + (1 - self.velocity_smoothing) * avg_vx
                        avg_vy = self.velocity_smoothing * prev_vy + (1 - self.velocity_smoothing) * avg_vy
                    
                    return avg_vx, avg_vy
        return 0.0, 0.0

    def predict_position(self, track_id):
        """Predict next position based on velocity"""
        if track_id in self.tracked_objects and track_id in self.velocities:
            current_center = self.tracked_objects[track_id]['center']
            vx, vy = self.velocities[track_id]
            
            # Predict position 1 frame ahead
            predicted_x = current_center[0] + vx * 0.033  # Assuming 30fps
            predicted_y = current_center[1] + vy * 0.033
            
            return (predicted_x, predicted_y)
        return None

    def calculate_detection_quality(self, detection):
        """Calculate quality score for detection"""
        confidence = detection['confidence']
        bbox = detection['bbox']
        area = bbox[2] * bbox[3]
        aspect_ratio = bbox[2] / max(bbox[3], 1)
        
        # Quality factors
        confidence_score = confidence
        size_score = min(area / 10000.0, 1.0)  # Normalize area
        aspect_score = 1.0 - abs(aspect_ratio - 0.7) / 2.0  # Prefer human-like ratios
        
        quality = (confidence_score * 0.5 + size_score * 0.3 + max(aspect_score, 0.2) * 0.2)
        return min(quality, 1.0)

    def update(self, detections):
        """Enhanced update with prediction and quality filtering"""
        # Filter low-quality detections
        quality_detections = []
        for detection in detections:
            quality = self.calculate_detection_quality(detection)
            if quality > self.confidence_threshold:
                detection['quality'] = quality
                quality_detections.append(detection)

        if len(quality_detections) == 0:
            # Mark all tracked objects as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.remove_track(track_id)
            return []

        # If no existing tracked objects, register all detections
        if len(self.tracked_objects) == 0:
            for detection in quality_detections:
                self.register_new_object(detection)
        else:
            # Enhanced distance calculation with prediction
            track_ids = list(self.tracked_objects.keys())
            detection_centers = []

            for detection in quality_detections:
                x, y, w, h = detection['bbox']
                center = (x + w // 2, y + h // 2)
                detection_centers.append(center)

            # Create enhanced distance matrix
            distance_matrix = np.zeros((len(track_ids), len(quality_detections)))
            for i, track_id in enumerate(track_ids):
                predicted_pos = self.predict_position(track_id)
                track_center = predicted_pos if predicted_pos else self.tracked_objects[track_id]['center']
                
                for j, det_center in enumerate(detection_centers):
                    # Euclidean distance with quality weighting
                    base_distance = math.sqrt(
                        (track_center[0] - det_center[0])**2 + 
                        (track_center[1] - det_center[1])**2
                    )
                    
                    # Weight by detection quality
                    quality_weight = quality_detections[j]['quality']
                    distance_matrix[i, j] = base_distance / max(quality_weight, 0.1)

            # Hungarian algorithm approximation
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            # Update existing tracks
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if distance_matrix[row, col] <= self.max_distance:
                    track_id = track_ids[row]
                    self.update_existing_track(track_id, quality_detections[col])
                    used_row_indices.add(row)
                    used_col_indices.add(col)

            # Handle unmatched detections and tracks
            unused_row_indices = set(range(len(track_ids))) - used_row_indices
            unused_col_indices = set(range(len(quality_detections))) - used_col_indices

            # Register new objects for unmatched high-quality detections
            for col in unused_col_indices:
                if quality_detections[col]['quality'] > 0.5:  # Only register high-quality new detections
                    self.register_new_object(quality_detections[col])

            # Mark unmatched tracks as disappeared
            for row in unused_row_indices:
                track_id = track_ids[row]
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1

                if self.disappeared[track_id] > self.max_disappeared:
                    self.remove_track(track_id)

        # Return enhanced detections with track IDs
        tracked_detections = []
        for track_id, obj_data in self.tracked_objects.items():
            detection = obj_data['detection'].copy()
            detection['track_id'] = track_id
            detection['is_locked'] = track_id in self.locked_objects
            detection['track_duration'] = time.time() - obj_data['first_seen']
            detection['stability'] = obj_data.get('stability', 0)

            # Add enhanced velocity information
            if track_id in self.velocities:
                detection['velocity_x'] = self.velocities[track_id][0]
                detection['velocity_y'] = self.velocities[track_id][1]
                detection['speed'] = math.sqrt(self.velocities[track_id][0]**2 + self.velocities[track_id][1]**2)
            else:
                detection['velocity_x'] = 0.0
                detection['velocity_y'] = 0.0
                detection['speed'] = 0.0

            # Add confidence stability
            if track_id in self.confidence_history:
                confidences = list(self.confidence_history[track_id])
                detection['avg_confidence'] = sum(confidences) / len(confidences)
                detection['confidence_stability'] = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0
            else:
                detection['avg_confidence'] = detection['confidence']
                detection['confidence_stability'] = 1.0

            tracked_detections.append(detection)

        return tracked_detections

    def register_new_object(self, detection):
        """Register a new object with enhanced initialization"""
        x, y, w, h = detection['bbox']
        center = (x + w // 2, y + h // 2)
        current_time = time.time()

        self.tracked_objects[self.next_track_id] = {
            'detection': detection,
            'center': center,
            'first_seen': current_time,
            'last_seen': current_time,
            'total_detections': 1,
            'stability': 0
        }

        # Initialize tracking data
        self.track_history[self.next_track_id].append({
            'center': center,
            'timestamp': current_time,
            'bbox': detection['bbox']
        })

        self.confidence_history[self.next_track_id].append(detection['confidence'])
        self.size_history[self.next_track_id].append(w * h)
        self.disappeared[self.next_track_id] = 0
        self.next_track_id += 1

    def update_existing_track(self, track_id, detection):
        """Update existing track with enhanced smoothing"""
        x, y, w, h = detection['bbox']
        new_center = (x + w // 2, y + h // 2)
        current_time = time.time()

        # Apply position smoothing
        old_center = self.tracked_objects[track_id]['center']
        smoothed_center = (
            self.position_smoothing * old_center[0] + (1 - self.position_smoothing) * new_center[0],
            self.position_smoothing * old_center[1] + (1 - self.position_smoothing) * new_center[1]
        )

        # Calculate velocity with smoothed position
        vx, vy = self.calculate_velocity(track_id, smoothed_center)
        self.velocities[track_id] = (vx, vy)

        # Update tracking data
        self.track_history[track_id].append({
            'center': smoothed_center,
            'timestamp': current_time,
            'bbox': detection['bbox']
        })

        self.confidence_history[track_id].append(detection['confidence'])
        self.size_history[track_id].append(w * h)

        # Update tracked object
        self.tracked_objects[track_id]['detection'] = detection
        self.tracked_objects[track_id]['center'] = smoothed_center
        self.tracked_objects[track_id]['last_seen'] = current_time
        self.tracked_objects[track_id]['total_detections'] += 1

        # Update stability score
        stability = min(self.tracked_objects[track_id]['total_detections'], self.stability_threshold)
        self.tracked_objects[track_id]['stability'] = stability

        # Reset disappeared counter
        self.disappeared[track_id] = 0

    def remove_track(self, track_id):
        """Remove a track that has disappeared"""
        if track_id in self.tracked_objects:
            del self.tracked_objects[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
        if track_id in self.locked_objects:
            self.locked_objects.remove(track_id)
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.velocities:
            del self.velocities[track_id]
        if track_id in self.confidence_history:
            del self.confidence_history[track_id]
        if track_id in self.size_history:
            del self.size_history[track_id]

    def lock_object(self, track_id, reason="manual_lock"):
        """Lock an object for detailed tracking"""
        if track_id in self.tracked_objects:
            self.locked_objects.add(track_id)
            return True
        return False

    def unlock_object(self, track_id):
        """Unlock an object"""
        if track_id in self.locked_objects:
            self.locked_objects.remove(track_id)
            return True
        return False

class AdvancedObjectDetector:
    def __init__(self):
        # Enhanced COCO class names with confidence thresholds
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]

        self.running = False
        self.tracker = EnhancedObjectTracker(max_disappeared=45, max_distance=120)
        self.selected_track_id = None
        self.auto_lock_enabled = True

        # Enhanced background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Multiple cascade classifiers for better accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

        # Enhanced YOLO setup
        self.yolo_net = None
        self.yolo_output_layers = None
        self.load_yolo_model()

        # Camera setup with enhanced parameters
        self.setup_camera()

        # Frame processing parameters
        self.frame_skip = 1  # Process every frame for better accuracy
        self.confidence_threshold = 0.4
        self.nms_threshold = 0.3

        print("Advanced real-time object detection initialized")

    def load_yolo_model(self):
        """Enhanced YOLO model loading with multiple model support"""
        model_configs = [
            ('yolov4.weights', 'yolov4.cfg'),
            ('yolov3.weights', 'yolov3.cfg'),
            ('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        ]

        for weights, config in model_configs:
            try:
                if os.path.exists(weights) and os.path.exists(config):
                    self.yolo_net = cv2.dnn.readNet(weights, config)
                    layer_names = self.yolo_net.getLayerNames()
                    unconnected = self.yolo_net.getUnconnectedOutLayers()
                    self.yolo_output_layers = [layer_names[i - 1] for i in unconnected.flatten()]
                    
                    # Set backend for better performance
                    self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    
                    print(f"YOLO model loaded: {weights}")
                    return
            except Exception as e:
                print(f"Failed to load {weights}: {e}")
                continue

        print("No YOLO models available, using alternative detection methods")

    def setup_camera(self):
        """Enhanced camera setup with optimal parameters"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            exit()

        # Enhanced camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
        self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    def detect_with_enhanced_yolo(self, frame):
        """Enhanced YOLO detection with better preprocessing"""
        detections = []
        if self.yolo_net is None:
            return detections

        height, width = frame.shape[:2]

        # Enhanced preprocessing
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (608, 608), (0, 0, 0), True, crop=False
        )
        self.yolo_net.setInput(blob)
        outputs = self.yolo_net.forward(self.yolo_output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Boundary validation
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = min(w, width - x)
                    h = min(h, height - y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Enhanced NMS
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                detection = {
                    'class': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else 'unknown',
                    'confidence': confidences[i],
                    'bbox': [x, y, w, h],
                    'detection_method': 'yolo_enhanced'
                }
                detections.append(detection)

        return detections

    def detect_with_enhanced_haar_cascades(self, frame):
        """Enhanced Haar cascade detection with multiple cascades"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)

        # Multiple face detection approaches
        face_detections = []
        
        # Frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in faces:
            face_detections.append((x, y, w, h, 0.85, 'frontal_face'))

        # Profile faces
        profiles = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in profiles:
            face_detections.append((x, y, w, h, 0.75, 'profile_face'))

        # Filter overlapping face detections
        filtered_faces = self.filter_cascade_overlaps(face_detections)
        
        for x, y, w, h, conf, method in filtered_faces:
            detection = {
                'class': 'person',
                'confidence': conf,
                'bbox': [int(x), int(y), int(w), int(h)],
                'detection_method': f'haar_{method}'
            }
            detections.append(detection)

        # Upper body detection if few face detections
        if len(filtered_faces) < 2:
            upper_bodies = self.upper_body_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 80)
            )
            for (x, y, w, h) in upper_bodies:
                # Avoid overlap with face detections
                if not self.overlaps_with_faces(x, y, w, h, filtered_faces):
                    detection = {
                        'class': 'person',
                        'confidence': 0.7,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'detection_method': 'haar_upper_body'
                    }
                    detections.append(detection)

        # Full body detection if very few detections
        if len(detections) < 1:
            bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 160)
            )
            for (x, y, w, h) in bodies:
                detection = {
                    'class': 'person',
                    'confidence': 0.65,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'detection_method': 'haar_full_body'
                }
                detections.append(detection)

        return detections

    def filter_cascade_overlaps(self, detections):
        """Filter overlapping cascade detections"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        detections.sort(key=lambda x: x[4], reverse=True)
        
        filtered = []
        for current in detections:
            x1, y1, w1, h1, conf1, method1 = current
            
            overlaps = False
            for existing in filtered:
                x2, y2, w2, h2, conf2, method2 = existing
                
                # Calculate overlap
                overlap_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                              max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                area1 = w1 * h1
                area2 = w2 * h2
                
                if overlap_area > 0.3 * min(area1, area2):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(current)
        
        return filtered

    def overlaps_with_faces(self, x, y, w, h, face_detections):
        """Check if detection overlaps with face detections"""
        for fx, fy, fw, fh, _, _ in face_detections:
            overlap_area = max(0, min(x + w, fx + fw) - max(x, fx)) * \
                          max(0, min(y + h, fy + fh) - max(y, fy))
            if overlap_area > 0.2 * (w * h):
                return True
        return False

    def detect_with_enhanced_background_subtraction(self, frame):
        """Enhanced background subtraction with improved filtering"""
        detections = []

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Enhanced morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        # Gaussian blur to smooth
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:  # Larger minimum area
                x, y, w, h = cv2.boundingRect(contour)

                # Enhanced filtering criteria
                aspect_ratio = float(w) / h
                extent = area / (w * h)
                solidity = area / cv2.contourArea(cv2.convexHull(contour))

                # More sophisticated filtering
                if (0.2 < aspect_ratio < 4.0 and 
                    w > 40 and h > 60 and 
                    extent > 0.3 and 
                    solidity > 0.5):
                    
                    # Calculate confidence based on movement characteristics
                    confidence = min(0.8, (area / 5000.0) * extent * solidity)
                    
                    detection = {
                        'class': 'moving_object',
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'detection_method': 'background_subtraction_enhanced'
                    }
                    detections.append(detection)

        return detections

    def detect_objects(self, frame):
        """Enhanced main detection function with intelligent method selection"""
        all_detections = []

        # Primary: Enhanced YOLO (most accurate)
        if self.yolo_net is not None:
            yolo_detections = self.detect_with_enhanced_yolo(frame)
            all_detections.extend(yolo_detections)

        # Secondary: Enhanced Haar cascades (reliable for people)
        haar_detections = self.detect_with_enhanced_haar_cascades(frame)
        all_detections.extend(haar_detections)

        # Tertiary: Enhanced background subtraction (motion-based)
        if len(all_detections) < 3:  # Only if few detections from other methods
            bg_detections = self.detect_with_enhanced_background_subtraction(frame)
            all_detections.extend(bg_detections)

        # Enhanced filtering with quality assessment
        filtered_detections = self.advanced_filter_detections(all_detections, frame)

        return filtered_detections

    def advanced_filter_detections(self, detections, frame):
        """Advanced detection filtering with quality assessment"""
        if not detections:
            return detections

        # Quality scoring for each detection
        scored_detections = []
        for detection in detections:
            quality_score = self.calculate_detection_quality_score(detection, frame)
            detection['quality_score'] = quality_score
            scored_detections.append(detection)

        # Sort by quality score
        scored_detections.sort(key=lambda x: x['quality_score'], reverse=True)

        # Advanced NMS with quality weighting
        boxes = []
        scores = []
        indices = []

        for i, detection in enumerate(scored_detections):
            x, y, w, h = detection['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(detection['quality_score'])
            indices.append(i)

        if len(boxes) == 0:
            return detections

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply enhanced NMS
        keep_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.2)

        filtered_detections = []
        if len(keep_indices) > 0:
            keep_indices = keep_indices.flatten()
            for i in keep_indices:
                if scored_detections[indices[i]]['quality_score'] > 0.3:
                    filtered_detections.append(scored_detections[indices[i]])

        return filtered_detections

    def calculate_detection_quality_score(self, detection, frame):
        """Calculate comprehensive quality score for detection"""
        confidence = detection['confidence']
        bbox = detection['bbox']
        x, y, w, h = bbox
        
        # Basic quality factors
        area = w * h
        aspect_ratio = w / max(h, 1)
        
        # Confidence score (40%)
        confidence_score = confidence
        
        # Size score (20%) - prefer medium to large objects
        optimal_area = 10000  # Approximate good detection size
        size_score = 1.0 - abs(area - optimal_area) / (optimal_area * 2)
        size_score = max(0, min(size_score, 1.0))
        
        # Aspect ratio score (15%) - prefer human-like ratios
        if detection['class'] == 'person':
            optimal_ratio = 0.5  # Height > width for people
            aspect_score = 1.0 - abs(aspect_ratio - optimal_ratio) / 2.0
        else:
            aspect_score = 0.7  # Neutral for other objects
        aspect_score = max(0, min(aspect_score, 1.0))
        
        # Position score (15%) - prefer center areas, penalize edges
        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = x + w//2, y + h//2
        dist_from_center = math.sqrt(
            ((center_x - frame_w//2) / (frame_w//2))**2 + 
            ((center_y - frame_h//2) / (frame_h//2))**2
        )
        position_score = 1.0 - min(dist_from_center, 1.0)
        
        # Method reliability score (10%)
        method_scores = {
            'yolo_enhanced': 1.0,
            'haar_frontal_face': 0.9,
            'haar_profile_face': 0.8,
            'haar_upper_body': 0.7,
            'haar_full_body': 0.6,
            'background_subtraction_enhanced': 0.5
        }
        method = detection.get('detection_method', 'unknown')
        method_score = method_scores.get(method, 0.4)
        
        # Calculate weighted quality score
        quality = (confidence_score * 0.4 + 
                  size_score * 0.2 + 
                  aspect_score * 0.15 + 
                  position_score * 0.15 + 
                  method_score * 0.1)
        
        return min(quality, 1.0)

    def save_detection_to_db(self, detection):
        """Enhanced database saving with error handling"""
        try:
            conn = sqlite3.connect('detections.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, 
                                      track_id, is_locked, track_duration, velocity_x, velocity_y)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                detection['class'],
                detection['confidence'],
                detection['bbox'][0],
                detection['bbox'][1],
                detection['bbox'][2],
                detection['bbox'][3],
                detection.get('track_id', None),
                detection.get('is_locked', False),
                detection.get('track_duration', 0.0),
                detection.get('velocity_x', 0.0),
                detection.get('velocity_y', 0.0)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")

    def draw_enhanced_detections(self, frame, detections):
        """Enhanced visualization with more information"""
        detection_colors = {
            'yolo_enhanced': (0, 255, 0),
            'haar_frontal_face': (255, 100, 0),
            'haar_profile_face': (255, 150, 0),
            'haar_upper_body': (255, 200, 0),
            'haar_full_body': (0, 0, 255),
            'background_subtraction_enhanced': (255, 255, 0)
        }

        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            method = detection.get('detection_method', 'unknown')
            track_id = detection.get('track_id', None)
            is_locked = detection.get('is_locked', False)
            track_duration = detection.get('track_duration', 0.0)
            velocity_x = detection.get('velocity_x', 0.0)
            velocity_y = detection.get('velocity_y', 0.0)
            speed = detection.get('speed', 0.0)
            stability = detection.get('stability', 0)
            quality_score = detection.get('quality_score', 0.0)

            # Enhanced color coding
            if is_locked:
                box_color = (0, 0, 255)  # Red for locked
                thickness = 4
            elif track_id == self.selected_track_id:
                box_color = (255, 255, 0)  # Yellow for selected
                thickness = 3
            elif stability >= 5:
                box_color = (0, 255, 0)  # Green for stable tracks
                thickness = 2
            else:
                box_color = detection_colors.get(method, (128, 128, 128))
                thickness = 2

            # Main bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)

            # Quality indicator bar
            quality_width = int(w * quality_score)
            cv2.rectangle(frame, (x, y - 8), (x + quality_width, y - 3), (0, 255, 0), -1)
            cv2.rectangle(frame, (x + quality_width, y - 8), (x + w, y - 3), (0, 0, 255), -1)

            # Track information panel
            if track_id is not None:
                # Track ID circle
                cv2.circle(frame, (x + 20, y + 20), 15, box_color, -1)
                cv2.putText(frame, f"T{track_id}", (x + 10, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Stability indicator
                stability_color = (0, 255, 0) if stability >= 5 else (255, 255, 0) if stability >= 2 else (255, 0, 0)
                cv2.rectangle(frame, (x + w - 30, y + 5), (x + w - 5, y + 15), stability_color, -1)
                cv2.putText(frame, f"S{stability}", (x + w - 28, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

                # Lock indicator
                if is_locked:
                    cv2.rectangle(frame, (x + w - 25, y + 20), (x + w - 5, y + 35), (0, 0, 255), -1)
                    cv2.putText(frame, "L", (x + w - 20, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Speed indicator for moving objects
                if speed > 5:
                    speed_text = f"{speed:.0f}"
                    cv2.putText(frame, speed_text, (x + w - 40, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Detection number
            cv2.circle(frame, (x + w - 15, y + 15), 12, box_color, -1)
            cv2.putText(frame, str(i + 1), (x + w - 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Enhanced trajectory visualization
            if track_id and track_id in self.tracker.track_history:
                history = list(self.tracker.track_history[track_id])
                if len(history) > 1:
                    points = []
                    for hist_item in history[-15:]:
                        hist_bbox = hist_item['bbox']
                        center = (hist_bbox[0] + hist_bbox[2] // 2, hist_bbox[1] + hist_bbox[3] // 2)
                        points.append(center)

                    # Draw trajectory with gradient
                    for j in range(1, len(points)):
                        alpha = j / len(points)
                        color_intensity = int(255 * alpha)
                        trajectory_color = (0, color_intensity, 0)
                        cv2.line(frame, points[j-1], points[j], trajectory_color, max(1, int(3 * alpha)))

            # Enhanced label with comprehensive info
            label = f"{class_name}: {confidence:.2f}"
            if track_id is not None:
                track_info = f"T{track_id} {track_duration:.1f}s S{stability}"
                if speed > 1:
                    track_info += f" {speed:.0f}px/s"
                quality_info = f"Q:{quality_score:.2f} {method.split('_')[0]}"
            else:
                track_info = f"Q:{quality_score:.2f}"
                quality_info = method.split('_')[0]

            # Dynamic label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            track_size = cv2.getTextSize(track_info, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0] if track_info else (0, 0)
            quality_size = cv2.getTextSize(quality_info, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            max_width = max(label_size[0], track_size[0], quality_size[0])

            cv2.rectangle(frame, (x, y - 45), (x + max_width + 10, y), box_color, -1)
            cv2.putText(frame, label, (x + 2, y - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if track_info:
                cv2.putText(frame, track_info, (x + 2, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(frame, quality_info, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Enhanced stats panel
        self.draw_enhanced_stats_panel(frame, detections)

        return frame

    def draw_enhanced_stats_panel(self, frame, detections):
        """Enhanced statistics panel with more detailed information"""
        stats_x = frame.shape[1] - 220
        stats_y = 30

        # Background with transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x - 10, stats_y - 10), (frame.shape[1] - 10, stats_y + 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (stats_x - 10, stats_y - 10), (frame.shape[1] - 10, stats_y + 140), (255, 255, 255), 2)

        # Enhanced statistics
        total_tracks = len(self.tracker.tracked_objects)
        locked_tracks = len(self.tracker.locked_objects)
        stable_tracks = sum(1 for obj in self.tracker.tracked_objects.values() if obj.get('stability', 0) >= 5)
        high_quality = sum(1 for det in detections if det.get('quality_score', 0) > 0.7)

        cv2.putText(frame, "DETECTION STATUS", (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Active Tracks: {total_tracks}", (stats_x, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(frame, f"Stable: {stable_tracks}", (stats_x, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cv2.putText(frame, f"Locked: {locked_tracks}", (stats_x, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(frame, f"High Quality: {high_quality}", (stats_x, stats_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cv2.putText(frame, f"Selected: {self.selected_track_id or 'None'}", (stats_x, stats_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        cv2.putText(frame, "CONTROLS:", (stats_x, stats_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, "L-Lock U-Unlock", (stats_x, stats_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        cv2.putText(frame, "1-9-Select Q-Quit", (stats_x, stats_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)

    def run_detection(self):
        """Enhanced main detection loop with better performance"""
        self.running = True
        frame_count = 0
        fps_counter = 0
        start_time = time.time()

        print("Starting advanced real-time object detection...")
        print("Enhanced features: Multi-method detection, quality scoring, stability tracking")
        print("Controls: 'q' to quit, 'l' to lock, 'u' to unlock, '1-9' to select track")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break

            frame_count += 1
            fps_counter += 1

            # Process every frame for maximum accuracy
            detections = self.detect_objects(frame)

            # Update tracker with enhanced detections
            tracked_detections = self.tracker.update(detections)

            # Enhanced reporting
            current_time = time.time()
            if current_time - start_time >= 1.0:  # Every second
                fps = fps_counter / (current_time - start_time)
                fps_counter = 0
                start_time = current_time

                if tracked_detections:
                    print(f"\nFPS: {fps:.1f} | Frame {frame_count}: {len(tracked_detections)} objects tracked")
                    stable_tracks = [d for d in tracked_detections if d.get('stability', 0) >= 5]
                    high_quality = [d for d in tracked_detections if d.get('quality_score', 0) > 0.7]
                    
                    print(f"  Stable tracks: {len(stable_tracks)}, High quality: {len(high_quality)}")
                    
                    for i, det in enumerate(tracked_detections[:3]):
                        quality = det.get('quality_score', 0)
                        stability = det.get('stability', 0)
                        speed = det.get('speed', 0)
                        print(f"  {i+1}. {det['class']} (conf: {det['confidence']:.2f}, q: {quality:.2f}) "
                              f"Track: {det.get('track_id', 'N/A')} (stab: {stability}, speed: {speed:.0f})")

                # Save high-quality detections to database
                for detection in tracked_detections:
                    if detection.get('quality_score', 0) > 0.5:
                        threading.Thread(target=self.save_detection_to_db, args=(detection,), daemon=True).start()

            # Enhanced visualization
            frame = self.draw_enhanced_detections(frame, tracked_detections)

            # Display frame
            cv2.imshow('Advanced Real-Time Object Detection', frame)

            # Enhanced key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                if self.selected_track_id:
                    self.tracker.lock_object(self.selected_track_id)
                    print(f"Locked track {self.selected_track_id}")
                elif tracked_detections:
                    best_detection = max(tracked_detections, key=lambda x: x.get('quality_score', 0))
                    track_id = best_detection.get('track_id')
                    if track_id:
                        self.tracker.lock_object(track_id)
                        print(f"Locked highest quality track {track_id}")
            elif key == ord('u'):
                if self.selected_track_id:
                    self.tracker.unlock_object(self.selected_track_id)
                    print(f"Unlocked track {self.selected_track_id}")
                else:
                    for track_id in list(self.tracker.locked_objects):
                        self.tracker.unlock_object(track_id)
                    print("Unlocked all objects")
            elif key >= ord('1') and key <= ord('9'):
                track_num = key - ord('1')
                if track_num < len(tracked_detections):
                    self.selected_track_id = tracked_detections[track_num].get('track_id')
                    print(f"Selected track {self.selected_track_id}")

        self.stop()

    def stop(self):
        """Enhanced cleanup"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Advanced detection system stopped")

def print_enhanced_detection_stats():
    """Enhanced statistics reporting"""
    try:
        conn = sqlite3.connect('detections.db')
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM detections')
        total = cursor.fetchone()[0]

        cursor.execute('''
            SELECT object_class, COUNT(*) as count, AVG(confidence) as avg_conf, 
                   AVG(track_duration) as avg_duration
            FROM detections 
            WHERE confidence > 0.5
            GROUP BY object_class 
            ORDER BY count DESC 
            LIMIT 10
        ''')

        print(f"\n=== Enhanced Detection Statistics (Total: {total}) ===")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]} detections (avg conf: {row[2]:.2f}, avg duration: {row[3]:.1f}s)")

        # Track quality statistics
        cursor.execute('''
            SELECT COUNT(DISTINCT track_id) as unique_tracks,
                   AVG(track_duration) as avg_track_time,
                   MAX(track_duration) as max_track_time
            FROM detections 
            WHERE track_id IS NOT NULL
        ''')
        
        track_stats = cursor.fetchone()
        if track_stats[0]:
            print(f"\nTracking Performance:")
            print(f"Unique tracks: {track_stats[0]}")
            print(f"Average track duration: {track_stats[1]:.1f}s")
            print(f"Longest track: {track_stats[2]:.1f}s")

        conn.close()
    except Exception as e:
        print(f"Error reading enhanced stats: {e}")

if __name__ == '__main__':
    # Initialize database
    init_db()

    print("Advanced Real-Time Object Detection System")
    print("==========================================")
    print("Enhanced Features:")
    print("- Multi-method detection (YOLO + Haar + Background)")
    print("- Quality scoring and filtering")
    print("- Enhanced object tracking with prediction")
    print("- Velocity and stability analysis")
    print("- Advanced visualization")
    print()

    detector = AdvancedObjectDetector()

    try:
        detector.run_detection()
    except KeyboardInterrupt:
        print("\nStopping detection...")
        detector.stop()
    finally:
        print_enhanced_detection_stats()
