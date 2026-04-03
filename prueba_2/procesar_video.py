import cv2
from ultralytics import YOLO

# 1. Cargamos el modelo (se asegura de usar el nano para que tu netbook no sufra)
model = YOLO("yolo11n.pt") 

# 2. INDICÁ EL NOMBRE DE TU VIDEO AQUÍ
# El video tiene que estar en la misma carpeta que este script
video_path = "proyecto.mp4" 
cap = cv2.VideoCapture(video_path)

# Verificamos si el video se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video. Revisá el nombre del archivo.")
    exit()

print("Procesando video... Presioná 'q' para detener.")

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # La IA analiza el fotograma del video
        # Usamos stream=True para que consuma menos memoria RAM
        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()
            # Mostramos el video con las detecciones
            cv2.imshow("Análisis de Video - Proyecto Córdoba", annotated_frame)

        # Si presionás 'q', se sale del proceso
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # El video terminó
        break

cap.release()
cv2.destroyAllWindows()
print("Proceso finalizado.")