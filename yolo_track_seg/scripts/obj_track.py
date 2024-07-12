from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# to include a tracker, need to include a.yaml tracker file as an 
results = model.track(source = "videos/people_walking1.mp4", show=True)  # no tracker included, so tracking with default tracker