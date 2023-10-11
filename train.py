from ultralytics import YOLO
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8m.yaml').load('yolov8m.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/kaggle/working/yolov8/mydata/mydata_kaggle.yaml', epochs=100, imgsz=640, batch=32, device=[0,1], pretrained=True, amp=False, val=True)
