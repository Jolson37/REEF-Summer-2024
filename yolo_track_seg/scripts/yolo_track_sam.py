import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Set the model and device
model = YOLO("models/yolov8n.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Open the video file
video_path = "videos/grass_parking_lot_pan.mp4" #"videos/drone_lot_pan_fast.mp4" #"videos/people_walking1.mp4"
cap = cv2.VideoCapture(video_path)

# Get the start time
time_start = time.time()

# Frame Size
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))

# Store the track history
track_history = defaultdict(lambda: [])

################################################################################################### SAM Init
# Select sam model and model type, choose (from fast to high fidelity): "vit_b", "vit_h"
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# Use the SamPredictor (alternate is SamAutomaticMaskGenerator which is non-specific)
predictor = SamPredictor(sam)

# Function to show the generated masks in a plot (From Internet)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

################################################################################################### End SAM Init
# Start a count for any intermittent processes that may need to be run (at certain frame numbers)
count = 0

# Loop through the video frames
while cap.isOpened():
    count = count + 1
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf = 0.3, iou = 0.5, device = device, persist=True, classes = [2])


        if results[0].boxes is None or results[0].boxes.id is None:
            pass

        # Get the boxes, track ids, and classifications for each detection in the frame
        else:
            boxes = results[0].boxes.xywh.cuda()
            track_ids = results[0].boxes.id.int().cuda().tolist()
            cls = results[0].boxes.cls.cuda()
            seg_boxes = results[0].boxes.xyxy.cuda()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # test_frame = results[0].plot(boxes = True, conf = True, labels = True)

        # Plot the tracks
        for box, track_id, c in zip(boxes, track_ids, cls):
                
                # Save the label for the box and add it to the dict with the track id and label
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
            
        # Every 12 frames segment the image once
        if count%12 == 0 and count !=0:
            # plot_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_boxes = seg_boxes
            predictor.set_image(image)
            
            # call predictor instead of generate to determine where you want masks to be captured
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (frame_height,frame_width))
            masks, scores, logits = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
            masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W

            # outputs all of the boxes that were made along with each corresponding mask generated
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
            plt.axis('on')
            # plt.savefig("masks_unlabeled/frame%d.jpg" % count)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Save the image as a jpg
        # cv2.imwrite("seg_class_images/frame%d.jpg" % count, test_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    else:
        # Break the loop if the end of the video is reached
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

