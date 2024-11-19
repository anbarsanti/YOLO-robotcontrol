"""
﷽
author: @anbarsanti
"""


from ultralytics import YOLO
import torch

# # Load a pretrained YOLO model
# model = YOLO("yolo8n-obb.pt") # Load a pre-trained model

# Check if a CUDA-compatible GPU is available on your system:
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load a model from scratch
model11 = YOLO("yolo11n-obb.yaml")

# Train the model using the '.yaml' dataset for a number of epochs
train_results = model11.train(
    data="model/data.yaml", # path to dataset YAML
    epochs = 100, # number of training epochs
    imgsz=640, # training image size
    device=0, # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    multi_scale=True,
)

# Evaluate the model's performance on the validation set
metrics = model11.val(data = "model/data.yaml")

# Save the model
# model11.save("yolo11-obb-11-15.pt")

# PERFORM OBJECT DETECTION ON AN IMAGE USING THE MODEL
# results = model.predict("dataset/test/images", save=True)