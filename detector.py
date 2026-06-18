import cv2
import os
import csv
from datetime import datetime

# Crear carpeta de capturas
if not os.path.exists("capturas"):
    os.makedirs("capturas")

# Archivo de registro
log_file = "capturas/log.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["fecha", "hora", "rostros"])

# Detector de rostros
clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Cámara
camara = cv2.VideoCapture(0)

ultimo_guardado = 0
intervalo = 5  # segundos

while True:

    ret, frame = camara.read()

    if not ret:
        break

    gris = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    rostros = clasificador.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=6
    )

    # Dibujar rostros
    for (x, y, w, h) in rostros:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Contador de rostros
    cv2.putText(
        frame,
        f"Rostros: {len(rostros)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Fecha y hora
    fecha_hora = datetime.now().strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    cv2.putText(
        frame,
        fecha_hora,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Guardar evidencia
    ahora = datetime.now().timestamp()

    if len(rostros) > 0 and (
        ahora - ultimo_guardado
    ) > intervalo:

        nombre = (
            f"capturas/face_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        cv2.imwrite(
            nombre,
            frame
        )

        with open(
            log_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                len(rostros)
            ])

        print(
            f"[INFO] Captura guardada: {nombre}"
        )

        ultimo_guardado = ahora

    cv2.imshow(
        "SMART FACE SECURITY SYSTEM",
        frame
    )

    # ESC para salir
    if cv2.waitKey(1) == 27:
        break

camara.release()
cv2.destroyAllWindows()