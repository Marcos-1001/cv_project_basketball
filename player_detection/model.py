# %% 
from roboflow import Roboflow
from ultralytics import YOLO

# %% 
rf = Roboflow(api_key="RS4X51uQQQeLtunKB3bO")
project = rf.workspace("roboflow-universe-projects").project("basketball-players-fy4c2")
version = project.version(25)
dataset = version.download("yolov11")


# %%                 
model = YOLO("yolo11n.pt")  # Load a pre-trained YOLOv1 nano model
results = model.train(data=dataset.location+'/data.yaml', epochs=150,
                      )

# Save the trained model
model.save("yolo11n-trained-25.pt")
