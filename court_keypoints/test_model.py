import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Carga modelo pose
model = YOLO("yolov8n-pose-trained.pt")  # Use GPU if available


# Define vídeo
cap = cv2.VideoCapture("../testing_video/nba_20_min.mp4")
if not cap.isOpened():
    raise RuntimeError("No se puede abrir el video")

    # Obtener todas las clases del modelo
all_classes = model.names
print("Clases del modelo:", all_classes)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.6, show=False)
    out = frame.copy()

    for pose in results:
        # Obtener clase del objeto (ej. basketball)
        if pose.boxes is not None and len(pose.boxes.cls) > 0:
            cls_id = int(pose.boxes.cls[0].item())
            cls_name = pose.names[cls_id]
        else:
            cls_name = "Unknown"

        print(cls_name)
        try: 
            # Dibujar etiqueta de clase
            box = pose.boxes.xyxy[0].cpu().numpy().astype(int)
            cv2.putText(out, cls_name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Extraer keypoints y fiabilidad
            kpts_xy = pose.keypoints.xy
            kpts_conf = pose.keypoints.conf

            for person_kpts, person_conf in zip(kpts_xy, kpts_conf):
                pts = person_kpts.cpu().numpy()   # shape (N,2)
                confs = person_conf.cpu().numpy() # shape (N,)

                # Dibujar keypoints
                for (x, y), c in zip(pts, confs):
                    if c > 0.5:
                        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
        except Exception as e:
            print(f"Error procesando pose: {e}")
    # Mostrar con Matplotlib
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.000001)

cap.release()
cv2.destroyAllWindows()