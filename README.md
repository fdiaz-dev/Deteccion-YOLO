# 🚀 Detección de Objetos con YOLOv11 - Córdoba, Argentina

Sistema de visión artificial desarrollado para la detección y clasificación automática de objetos urbanos. Este proyecto evolucionó desde pruebas básicas con webcam hasta un sistema de recolección de datos optimizado para hardware de recursos limitados (Netbook Juana Manso).

## 📈 Evolución del Proyecto

El desarrollo se dividió en etapas para mejorar el rendimiento y la organización de los datos:

1.  **Etapa 1 (prueba_1):** Detección en tiempo real utilizando la cámara integrada del equipo.
2.  **Etapa 2 (prueba_2):** Procesamiento de archivos de video (.mp4) con visualización de cuadros (bounding boxes) en pantalla.
3.  **Etapa 3 (prueba_3_automatizado):** Implementación de lógica para crear carpetas automáticas por cada clase detectada y guardar capturas de los objetos.
4.  **Versión Actual (recolector_sin_repetidos.py):** * **Optimización de recursos:** Se eliminó la ventana de visualización (`cv2.imshow`) para aumentar los FPS y reducir el consumo de CPU/RAM.
    * **Gestión de archivos:** El script identifica el objeto, crea una carpeta con su nombre y guarda la captura solo si es un objeto relevante, evitando duplicados innecesarios.

## 🛠️ Tecnologías utilizadas
* **Python 3.x**
* **Ultralytics (YOLOv11n)**
* **OpenCV** (Procesamiento de imágenes)

## 💻 Instalación y Uso

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/fdiaz-dev/Deteccion-YOLO.git](https://github.com/fdiaz-dev/Deteccion-YOLO.git)
