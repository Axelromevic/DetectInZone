import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Clase para procesar los frames del video
class IntrusionDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.intrusion_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Procesamiento del frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        area_pts = np.array([[240, 320], [480, 320], [640, img.shape[0]], [50, img.shape[0]]])
        imAux = np.zeros_like(gray)
        cv2.drawContours(imAux, [area_pts], -1, (255), -1)
        image_area = cv2.bitwise_and(gray, gray, mask=imAux)

        # Detector de movimiento
        fgmask = self.fgbg.apply(image_area)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        # Detección de intrusión
        cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.intrusion_count += 1

        # Mostrar el contador de intrusiones
        intrusion_text = f"Intrusion: {self.intrusion_count}"
        cv2.putText(img, intrusion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

# Interfaz de usuario con Streamlit
st.title("Intrusion Detection with Streamlit WebRTC")

st.write("Este proyecto detecta intrusiones en un área específica usando Streamlit WebRTC.")

# Inicializar el streamer
webrtc_streamer(key="intrusion-detection", video_transformer_factory=IntrusionDetectionTransformer)
