import cv2
from ultralytics import YOLO

# 1. Cargamos el modelo YOLO26 nano (el más rápido para tu netbook)
model = YOLO("yolo11n.pt") 

# 2. Iniciamos la cámara (0 suele ser la de la netbook)
cap = cv2.VideoCapture(0)

print("Iniciando detección... Presioná 'q' para salir")

while cap.isOpened():
    # Leemos un cuadro de la cámara
    success, frame = cap.read()

    if success:
        # La IA analiza qué hay en la imagen
        results = model(frame)

        # Dibujamos los resultados (cuadraditos y nombres)
        annotated_frame = results[0].plot()

        # Mostramos la ventana con el video
        cv2.imshow("Detección en Vivo - Córdoba", annotated_frame)

        # Si presionás la tecla 'q', el programa se cierra
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Al terminar, liberamos la cámara y cerramos ventanas
cap.release()
cv2.destroyAllWindows()