import cv2
import os
from ultralytics import YOLO

# 1. Cargamos el modelo
model = YOLO("yolo11n.pt") 

# 2. Configuración del video y saltos
video_path = "proyecto.mp4" # <--- CAMBIÁ ESTO por el nombre de tu video
cap = cv2.VideoCapture(video_path)

# Para no guardar 30 fotos por segundo del mismo auto:
skip_frames = 30  # Guarda una foto cada 30 cuadros (aprox. cada 1 segundo)
frame_count = 0

print("Iniciando recolección automática...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Solo procesamos si toca el cuadro según el salto
    if frame_count % skip_frames == 0:
        results = model(frame)

        for r in results:
            for box in r.boxes:
                # Obtenemos el nombre de la clase (car, person, etc.)
                cls_name = model.names[int(box.cls[0])]
                conf = box.conf[0] # Nivel de confianza

                # Solo guardamos si la IA está segura (más del 50%)
                if conf > 0.5:
                    # Crear carpeta si no existe
                    if not os.path.exists(cls_name):
                        os.makedirs(cls_name)

                    # Recortar el objeto detectado
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]

                    # Guardar la imagen con un nombre único basado en el tiempo
                    img_name = f"{cls_name}/{cls_name}_{cv2.getTickCount()}.jpg"
                    cv2.imwrite(img_name, crop)
                    print(f"Guardado: {img_name}")

    frame_count += 1

cap.release()
print("¡Listo! Revisá las carpetas que se crearon en tu proyecto.")