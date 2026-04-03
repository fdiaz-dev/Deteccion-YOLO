import cv2
import os
from ultralytics import YOLO

# 1. Cargamos el modelo
model = YOLO("yolo11n.pt") 

video_path = "proyecto.mp4"
cap = cv2.VideoCapture(video_path)

ids_guardados = set()

print("Procesando video en segundo plano... Por favor, espera.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Ejecutamos el tracking (sin mostrar nada en pantalla)
    # verbose=False hace que la terminal no se llene de texto innecesario
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()
        clases = results[0].boxes.cls.int().cpu().tolist()

        for box, obj_id, cls in zip(boxes, ids, clases):
            if obj_id not in ids_guardados:
                cls_name = model.names[cls]
                
                if not os.path.exists(cls_name):
                    os.makedirs(cls_name)

                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]

                img_name = f"{cls_name}/{cls_name}_id_{obj_id}.jpg"
                cv2.imwrite(img_name, crop)
                
                ids_guardados.add(obj_id)
                # Solo imprimimos un mensaje corto para saber que está trabajando
                print(f"Capturado: {cls_name} ID:{obj_id}")

cap.release()
# Ya no necesitamos cv2.destroyAllWindows() porque nunca abrimos ninguna
print("¡Proceso completado con éxito!")