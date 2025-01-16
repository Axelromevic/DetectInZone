import streamlit as st
import numpy as np
import cv2
import time


# Inicialización de Streamlit
st.title("Intrusion Detection")
st.sidebar.title("Controls")

# Botones de control
pause = st.sidebar.button("Pause/Play")
quit_app = st.sidebar.button("Close")

# Estado Persistente
if 'pause_flag' not in st.session_state:
    st.session_state.pause_flag = False
if 'intrusion_count' not in st.session_state:
    st.session_state.intrusion_count = 0
if 'frame_pos' not in st.session_state:
    st.session_state.frame_pos = 0
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None  # Para guardar el último frame visible
if 'active_centroids' not in st.session_state:
    st.session_state.active_centroids = []  # Guardar los centroides activos

# Botón Pause/Play
if pause:
    st.session_state.pause_flag = not st.session_state.pause_flag

# Cerrar app
if quit_app:
    st.stop()

# Video
cap = cv2.VideoCapture('aeropuerto.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)  # Ir al frame actual

# Procesamiento del video
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
frame_container = st.empty()
start_time = time.time() - st.session_state.frame_pos / cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    # Si está en pausa, mostrar el último frame y saltar el resto del bucle
    if st.session_state.pause_flag:
        if st.session_state.last_frame is not None:
            frame_container.image(st.session_state.last_frame, channels="RGB", use_column_width=True)
        time.sleep(0.1)
        continue

    # Leer el frame
    ret, frame = cap.read()
    if not ret:
        st.write("Finished")
        break

    # Actualizar la posición del frame
    st.session_state.frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Procesamiento del frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    area_pts = np.array([[240, 320], [480, 320], [640, frame.shape[0]], [50, frame.shape[0]]])
    imAux = np.zeros_like(gray)
    cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    # Detector de movimiento
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Detección de intrusión
    new_centroids = []
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            new_centroids.append((cx, cy))

            # Incrementar el contador si es un nuevo centroide
            if all(np.linalg.norm(np.array((cx, cy)) - np.array(active)) > 30 for active in st.session_state.active_centroids):
                st.session_state.intrusion_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Actualizar centroides activos
    st.session_state.active_centroids = new_centroids

    # Cambiar color del contorno del área
    color_area = (0, 0, 255) if len(new_centroids) > 0 else (0, 255, 0)
    cv2.drawContours(frame, [area_pts], -1, color_area, 2)

    # Tiempo y contador de intrusiones
    elapsed_time = int(st.session_state.frame_pos / cap.get(cv2.CAP_PROP_FPS))
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_text = f"Time: {hours:02}:{minutes:02}:{seconds:02}"
    cv2.putText(frame, time_text, (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    intrusion_text = f"Intrusion: {st.session_state.intrusion_count}"
    cv2.putText(frame, intrusion_text, (frame.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convertir frame a RGB y mostrarlo en Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame = frame  # Guardar el último frame visible
    frame_container.image(frame, channels="RGB", use_column_width=True)

cap.release()
cv2.destroyAllWindows()
