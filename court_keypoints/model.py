# %% 
from roboflow import Roboflow
from ultralytics import YOLO

# %% 

rf = Roboflow(api_key="VeMJ4jaEsMPQWIbgTPri")
project = rf.workspace("ml-zgvkm").project("reloc2-den7l-nljsj")
version = project.version(1)
dataset = version.download("yolov8")
                
# %%                 
model = YOLO("yolov8n-pose.pt")  # Load a pre-trained YOLOv8 nano model
results = model.train(data=dataset.location+'/data.yaml', epochs=100)

# Save the trained model
model.save("yolov8n-pose-trained.pt")

# %% 

model = YOLO("yolov8n-pose-trained.pt")  # Load the trained model
results = model.predict(source=dataset.location+'/test/images', conf=0.55, show=True)
# Save the results
results.save(save_dir="predictions")