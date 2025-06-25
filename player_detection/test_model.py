import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Carga el modelo pose (pre-entrenado en COCO, detecta personas)
model = YOLO("yolo11n-trained.pt")



# Abre el video
cap = cv2.VideoCapture("../testing_video/nba_20_min.mp4")
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

# Pausa mínima para Matplotlib
pause_time = 0.0000001
print("Clases del modelo:", model.names)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(frame.shape)
    exit()
    # Inferencia
    results = model.track(source=frame, conf=0.2, show=False,
                          iou=0.5, persist=True, tracker="botsort.yaml")
    
    out = frame.copy()

    for res in results:
        boxes = res.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = res.names[cls_id]
            if cls_name != "Player" and cls_name != "Ball":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # Etiqueta y caja
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar con OpenCV
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.pause(pause_time)
    plt.clf()
    # Mostrar con OpenCV
    #cv2.imshow("Detección de Jugadores y Balón", out)
    #cv2.waitKey(1)  # Espera 1 ms para permitir la actualización de la ventana
    
    

cap.release()
cv2.destroyAllWindows()
