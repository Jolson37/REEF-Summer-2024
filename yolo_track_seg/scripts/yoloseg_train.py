from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8x-seg.yaml")  # build a new model from YAML
model = YOLO("yolov8x-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Set the device
device = 'cuda:0,1,2,3'

# Train the model (paramse: change epochs for training loops, imgsz = 640 (maximum), batch = multiple of (# of gpus), device = device, name the model)
results = model.train(data="datasets/segmentation_datasets/IR/IR_FINAL_DEMO/data.yaml", epochs=100, imgsz=640, batch = 16, device = device, name = 'yolov8x-seg_BIG')
