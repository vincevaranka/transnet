import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker

# Load the pre-trained model for object detection
protopath = "lib/MobileNetSSD_deploy.prototxt"
modelpath = "lib/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Class labels for the MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

# Initialize counts
total_up = 0
total_down = 0

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

def overlay_logo(frame, logo):
    frame_h, frame_w = frame.shape[:2]
    logo_h, logo_w = logo.shape[:2]
    if logo_w > frame_w // 4:
        scale_ratio = (frame_w // 4) / logo_w
        logo = cv2.resize(logo, (int(logo_w * scale_ratio), int(logo_h * scale_ratio)))
        logo_h, logo_w = logo.shape[:2]
    x_offset = frame_w - logo_w - 10
    y_offset = frame_h - logo_h - 10
    roi = frame[y_offset:y_offset + logo_h, x_offset:x_offset + logo_w]
    gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_logo, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    logo_fg = cv2.bitwise_and(logo, logo, mask=mask)
    dst = cv2.add(frame_bg, logo_fg)
    frame[y_offset:y_offset + logo_h, x_offset:x_offset + logo_w] = dst
    return frame

def main():
    global total_up, total_down

    cap = cv2.VideoCapture('source/source.mp4')
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Video opened successfully.")

    total_frames = 0
    previous_centroids = {}
    line_position = 300
    frame_skip = 3
    logo = cv2.imread("source/logo.png")
    if logo is None:
        print("Error: Could not load logo image.")
        return

    print("Starting video processing...")

    # Get the width, height, and fps of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Removing VideoWriter initialization as we don't want to save the output
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        # Debugging information for frame reading
        if not ret:
            print("End of video or error in reading frames. Current total frames:", total_frames)
            break

        # Skip frames for performance
        total_frames += 1
        if total_frames % frame_skip != 0:
            continue

        frame = imutils.resize(frame, width=1024)
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()

        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.6:
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")
                    headY = int(startY + (endY - startY) / 3)
                    rects.append([startX, startY, endX, headY])

        boundingboxes = np.array(rects).astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

            if objectId in previous_centroids:
                prev_centroid = previous_centroids[objectId]
                if prev_centroid[1] < line_position and centroid[1] >= line_position:
                    total_down += 1
                elif prev_centroid[1] > line_position and centroid[1] <= line_position:
                    total_up += 1

            previous_centroids[objectId] = centroid

        cv2.line(frame, (0, line_position), (W, line_position), (255, 0, 0), 2)
        cv2.putText(frame, f"Up: {total_up}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Down: {total_down}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay the logo on the frame
        frame = overlay_logo(frame, logo)

        # Display the frame
        cv2.imshow("Person Head Counter", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
