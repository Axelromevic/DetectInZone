import streamlit as st
import numpy as np
import cv2
import time

# Inicialización de Streamlit
st.title("Intrusion Detection with Defined Area")
st.sidebar.title("Controls")

# Botones de control
pause = st.sidebar.button("Pause/Play")
quit_app = st.sidebar.button("Close")

# Estado Persistente
if 'pause_flag' not in st.session_state:
    st.session_state.pause_flag = False
if 'intrusion_count' not in st.session_state:
    st.session_state.intrusion_count = 0
if 'active_centroids' not in st.session_state:
    st.session_state.active_centroids = []

# Botón Pause/Play
if pause:
    st.session_state.pause_flag = not st.session_state.pause_flag

# Cerrar app
if quit_app:
    st.stop()

# Cargar el video
cap = cv2.VideoCapture('aeropuerto2.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
frame_container = st.empty()

# Obtener la tasa de cuadros por segundo del video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1 / fps  # Duración de cada cuadro en segundos

while cap.isOpened():
    if st.session_state.pause_flag:
        time.sleep(0.1)  # Pausa breve para evitar que el bucle corra sin control
        continue

    ret, frame = cap.read()
    if not ret:
        st.write("Finished")
        break

    # Definir área específica de detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    area_pts = np.array([[240, 320], [480, 320], [640, frame.shape[0]], [50, frame.shape[0]]])
    imAux = np.zeros_like(gray)
    cv2.drawContours(imAux, [area_pts], -1, (255), -1)  # Dibuja el área en blanco
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)  # Aplica la máscara

    # Detector de movimiento
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Detección de contornos
    new_centroids = []
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:  # Filtro por tamaño
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            new_centroids.append((cx, cy))

            # Incrementar el contador de intrusiones si el centroide es nuevo
            if all(np.linalg.norm(np.array((cx, cy)) - np.array(active)) > 30 for active in st.session_state.active_centroids):
                st.session_state.intrusion_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Actualizar centroides activos
    st.session_state.active_centroids = new_centroids

    # Dibujar el área de detección en el video
    cv2.drawContours(frame, [area_pts], -1, (0, 255, 0), 2)

    # Mostrar tiempo transcurrido
    elapsed_time = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_text = f"Time: {hours:02}:{minutes:02}:{seconds:02}"
    cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar contador de intrusiones
    intrusion_text = f"Intrusion: {st.session_state.intrusion_count}"
    cv2.putText(frame, intrusion_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar el frame procesado
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_container.image(frame, channels="RGB", use_container_width=True)

    # Ajustar velocidad del video
    time.sleep(frame_duration)

cap.release()
cv2.destroyAllWindows()
