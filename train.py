from ultralytics import YOLO

# # Load a pretrained YOLO model
# model = YOLO("yolo8n-obb.pt") # Load a pre-trained model

# Load a model from scratch
model11 = YOLO("yolo11n-obb.yaml")

# Train the model using the '.yaml' dataset for a number of epochs
train_results = model11.train(
    data="data.yaml", # path to dataset YAML
    epochs = 50, # number of training epochs
    imgsz=640, # training image size
    device="cpu", # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate the model's performance on the validation set
metrics = model11.val(data = "data.yaml")

# Save the model
model11.save('yolo11-obb-10-17.pt')

# PERFORM OBJECT DETECTION ON AN IMAGE USING THE MODEL
# results = model.predict("dataset/test/images", save=True)