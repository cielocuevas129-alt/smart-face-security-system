import cv2
import os
import csv
from datetime import datetime

# ==========================
# CONFIGURACIÓN
# ==========================

if not os.path.exists("capturas"):
    os.makedirs("capturas")

log_file = "capturas/log.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "fecha",
            "hora",
            "rostros"
        ])

# ==========================
# DETECTOR
# ==========================

clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camara = cv2.VideoCapture(0)

ultimo_guardado = 0
intervalo = 5

# ==========================
# BUCLE PRINCIPAL
# ==========================

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

    # Contador rostros
    cv2.putText(
        frame,
        f"Rostros: {len(rostros)}",
        (10, 35),
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
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Estado sistema
    cv2.putText(
        frame,
        "SISTEMA ACTIVO",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Total capturas
    total_capturas = len([
        f for f in os.listdir("capturas")
        if f.endswith(".jpg")
    ])

    cv2.putText(
        frame,
        f"Capturas: {total_capturas}",
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 200, 0),
        2
    )

    # Guardar evidencia
    ahora = datetime.now().timestamp()

    if len(rostros) > 0 and (
        ahora - ultimo_guardado
    ) > intervalo:

        nombre = (
            "capturas/face_" +
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            ) +
            ".jpg"
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
                datetime.now().strftime(
                    "%Y-%m-%d"
                ),
                datetime.now().strftime(
                    "%H:%M:%S"
                ),
                len(rostros)
            ])

        print(
            f"[INFO] Captura guardada: {nombre}"
        )

        ultimo_guardado = ahora

    cv2.imshow(
        "SMART FACE SECURITY SYSTEM v2.0",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

camara.release()
cv2.destroyAllWindows()
