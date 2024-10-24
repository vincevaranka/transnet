import cv2
import imutils
import numpy as np
from centroidtracker import CentroidTracker

# Load the pre-trained MobileNet SSD model for person detection
protopath = "lib/MobileNetSSD_deploy.prototxt"
modelpath = "lib/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

# Load logo image
logo = cv2.imread('source/logo.png')
if logo is None:
    print("Error: Could not load logo image. Check the file path.")
    exit(1)  # Exit if logo can't be loaded

max_logo_width = 100  # Maximum width for the logo

# Resize logo while maintaining aspect ratio
(h, w) = logo.shape[:2]
aspect_ratio = h / w
if w > max_logo_width:
    new_width = max_logo_width
    new_height = int(new_width * aspect_ratio)
    logo = cv2.resize(logo, (new_width, new_height))

# Class labels for MobileNet SSD (only care about 'person')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize the CentroidTracker
tracker = CentroidTracker(maxDisappeared=40, maxDistance=90)

# Define the ROI coordinates
roi_start_x, roi_start_y = 100, 100  # Top-left corner of the ROI
roi_end_x, roi_end_y = 340, 340      # Bottom-right corner of the ROI

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def main():
    # Open video stream (use '0' for webcam or provide a video file path)
    cap = cv2.VideoCapture('source/crowd1.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Video opened successfully.")

    total_frames = 0
    frame_skip = 4  # Skip frames to improve performance

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error in reading frames.")
            break

        total_frames += 1
        if total_frames % frame_skip != 0:
            continue

        frame = imutils.resize(frame, width=640)
        (H, W) = frame.shape[:2]

        # Prepare the frame for person detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()

        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.6:  # Confidence threshold
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")
                    
                    # Focus on the head (top third of the person's bounding box)
                    headY = int(startY + (endY - startY) / 3)
                    rects.append([startX, startY, endX, headY])

        # Apply non-max suppression to eliminate duplicate boxes
        boundingboxes = np.array(rects).astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        # Update the tracker
        objects = tracker.update(rects)

        # Initialize counter for people within ROI
        count_within_roi = 0
        
        # Draw the ROI rectangle
        cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 0, 0), 2)
        
        # Get the current count of people in the frame
        current_people_count = len(objects)
        
        # Draw bounding boxes and centroids for each detected head
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {objectId}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Check if the bounding box is within the ROI
            if (x1 >= roi_start_x and x2 <= roi_end_x) and (y1 >= roi_start_y and y2 <= roi_end_y):
                count_within_roi += 1

        # Display the counts
        cv2.putText(frame, f"Total People Count: {current_people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"People in ROI: {count_within_roi}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay the logo on the bottom right
        logo_height, logo_width = logo.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        y_offset = frame_height - logo_height - 10  # 10 pixels from the bottom
        x_offset = frame_width - logo_width - 10    # 10 pixels from the right

        # Ensure logo does not exceed frame boundaries
        if y_offset >= 0 and x_offset >= 0:
            frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = logo

        # Show the video feed with detections
        cv2.imshow("Head Tracker with ROI", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
