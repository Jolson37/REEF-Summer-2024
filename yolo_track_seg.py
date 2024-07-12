import cv2
import csv
import time
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Default YOLO models (from fast to high fidelity): "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"  

# Import a custom YOLO model
# model = YOLO("trained_weights/1_Jul_dataset_runs/segment/yolov8l-seg_ac_seg_data/weights/best.pt")
# model = YOLO("trained_weights/comb_ac_dataset_weights/segment/yolov8l-seg_comb_ac_data4/weights/best.pt")
model = YOLO("runs/segment/yolov8x-seg_BIG2/weights/best.pt")#"trained_models/IR/yolov8n-seg_FINAL/weights/FINAL.pt")

# Set the device to a gpu or cpu ("cuda", "cpu", or use a specific gpu like "cuda:1")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Open a video file from a selected path and use cv2 video capture
# References: https://www.videvo.net/, https://www.youtube.com/watch?v=hg0JLFleYRY, https://www.youtube.com/watch?v=IdrTiLb8Xoo
#"videos/[Paul Briden] Warthog Battle Damage.mp4" #"videos/180607_B_005.mp4" #"videos/900-1_900-4809-PD2_preview.mp4" #"videos/su-25_cropped.mp4" "videos/180607_B_007.mp4"
# "videos/Boson_Capture_4.mp4"
video_path =  "videos/Boson_Capture_6.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history in a dict
track_history = defaultdict(lambda: [])

# Start a count for any intermittent processes that may need to be run (at certain frame numbers)
count = 0

# Get the start time
time_start = time.time()

# Classes for the aicraft damage dataset
# names: {0: 'Crack', 1: 'Dent', 2: 'Missing-head', 3: 'Paint-off', 4: 'Scratch'}

# Loop through the video frames
while cap.isOpened():

    # add 1 to the count to count frame number
    count = count + 1

    # Read a frame from the video
    success, frame = cap.read()

    # If there is a frame to read continue, success is a bool
    if success:

        # Run YOLOv8 tracking on the frame while persisting tracks between frames, adjust conf, iou, tracker (default to botsort), + more
        results = model.track(frame, conf = 0.3, iou = 0.1, device = device, retina_masks = False, persist=True, tracker = 'trackers/bytetrack.yaml') #, classes = [0]) # Can set the model to track certain classes
        
        # Create a frame with the segments and boxes, can adjust the bool params to hide boxes, conf, id, label
        annotated_frame = results[0].plot()
        
        # If the result has no detection pass
        if results[0].boxes is None or results[0].boxes.id is None:
            pass

        # Get the boxes, track ids, and classifications for each detection in the frame
        else:
            boxes = results[0].boxes.xywh.cuda()
            track_ids = results[0].boxes.id.int().cuda().tolist()
            cls = results[0].boxes.cls.cuda()

            # Save the boxes, ids, and time stamps to the a dict
            for box, track_id, c in zip(boxes, track_ids, cls):
                
                # Save the label for the box and add it to the dict with the track id
                label = results[0].names[int(c)]
                track = track_history[track_id,label]

                elapsed_time = time.time() - time_start

                # Get box locations and save to a list (x y center point, time stamp)
                x, y, w, h = box 
                track.append((float(x), float(y), elapsed_time))

                # # Draw tracking lines on the annotated image
                # track1 = track_history[track_id]
                # track1.append((float(x), float(y))) 
                # points = np.hstack(track1).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 230, 230), thickness=10)
                
        # Display the annotated frame using opencv
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Save the annotated image every 12 frames
        # if count%12 == 0:
        #     cv2.imwrite("yolo_track_seg_save/frame%d.jpg" % count, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    # Break the loop if the end of the video is reached
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Print the tracking history ids, values, and time stamps
for id in track_history.keys():
                vals = list(track_history[id])
                start_val = vals[0]
                end_val = vals[len(vals)-1]
                x_rng = abs(end_val[0]-start_val[0])
                y_rng = abs(end_val[1]-start_val[1])
                print(id, "init detect: ", start_val, "last detect: ", end_val)

# For further post processing, save the tracked points, ids, and time stamps to a csv
# with open("tracking_data/dict.csv", 'w') as csv_file:  
#     writer = csv.writer(csv_file)
#     for key, value in track_history.items():
#        writer.writerow([key, value])
