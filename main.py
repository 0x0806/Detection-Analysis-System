import cv2
import numpy as np
import time


class SimpleDroneDetector:

    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True)
        self.drone_cascade = None
        self.setup_detection()

    def setup_detection(self):
        """Setup simple detection parameters"""
        # Create a simple circular template for drone detection
        self.drone_template = self.create_drone_template()

    def create_drone_template(self):
        """Create a simple drone template"""
        template = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(template, (25, 25), 20, 255, 2)
        cv2.circle(template, (25, 25), 10, 255, -1)
        return template

    def detect_movement(self, frame):
        """Detect moving objects that could be drones"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (adjust these values based on your needs)
            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio (drones are usually roughly square/circular)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    detection = {
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'area': area,
                        'confidence': min(area / 1000, 1.0)
                    }
                    detections.append(detection)

        return detections, fg_mask

    def detect_shapes(self, frame):
        """Detect circular/drone-like shapes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use HoughCircles to detect circular objects
        circles = cv2.HoughCircles(gray,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1,
                                   minDist=50,
                                   param1=50,
                                   param2=30,
                                   minRadius=10,
                                   maxRadius=100)

        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detection = {
                    'bbox': (x - r, y - r, 2 * r, 2 * r),
                    'center': (x, y),
                    'radius': r,
                    'confidence': 0.7,
                    'type': 'circular'
                }
                detections.append(detection)

        return detections

    def draw_detections(self, frame, detections, detection_type="motion"):
        """Draw detection results on frame"""
        display_frame = frame.copy()

        # Add title
        cv2.putText(display_frame,
                    f"Simple Drone Detector - {detection_type.title()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Detections: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection.get('confidence', 0)

            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.4:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence

            # Draw bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

            # Draw center point
            cv2.circle(display_frame, center, 5, color, -1)

            # Draw confidence
            cv2.putText(display_frame, f"Drone {i+1}: {confidence:.2f}",
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

            # Draw detection type if available
            det_type = detection.get('type', detection_type)
            cv2.putText(display_frame, det_type,
                        (bbox[0], bbox[1] + bbox[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return display_frame

    def run(self):
        """Run the simple drone detection system"""
        print("🚁 Simple Drone Detection System")
        print("=" * 40)
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle motion detection")
        print("  's' - Toggle shape detection")
        print("  'b' - Show background mask")
        print("=" * 40)

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return

        # Detection modes
        motion_detection = True
        shape_detection = False
        show_mask = False

        fps_counter = 0
        fps_start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Could not read frame")
                    break

                # Resize frame for better performance
                frame = cv2.resize(frame, (640, 480))

                all_detections = []
                mask = None

                # Motion-based detection
                if motion_detection:
                    motion_detections, mask = self.detect_movement(frame)
                    all_detections.extend(motion_detections)

                # Shape-based detection
                if shape_detection:
                    shape_detections = self.detect_shapes(frame)
                    all_detections.extend(shape_detections)

                # Draw results
                detection_type = []
                if motion_detection:
                    detection_type.append("motion")
                if shape_detection:
                    detection_type.append("shape")

                display_frame = self.draw_detections(
                    frame, all_detections, "+".join(detection_type) or "none")

                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    cv2.putText(display_frame, f"FPS: {fps:.1f}",
                                (10, display_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                2)

                # Show frames
                cv2.imshow('Simple Drone Detection', display_frame)

                if show_mask and mask is not None:
                    cv2.imshow('Motion Mask', mask)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    motion_detection = not motion_detection
                    print(
                        f"Motion detection: {'ON' if motion_detection else 'OFF'}"
                    )
                elif key == ord('s'):
                    shape_detection = not shape_detection
                    print(
                        f"Shape detection: {'ON' if shape_detection else 'OFF'}"
                    )
                elif key == ord('b'):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow('Motion Mask')
                    print(f"Background mask: {'ON' if show_mask else 'OFF'}")

        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    detector = SimpleDroneDetector()
    detector.run()


if __name__ == "__main__":
    main()
