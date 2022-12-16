from datetime import datetime
import imutils
import cv2
import numpy as np
from openvino.inference_engine import IECore

model_bin = "./models/face-detection-retail-0005.bin"
model_xml = "./models/face-detection-retail-0005.xml"
video_patch = "./video/Driver_1_Face_Cam.mp4"
fps = 0
initial_dt = datetime.now()
initial_ts = int(datetime.timestamp(initial_dt))


#1 Obtener el frame
def get_frame():
  vidcap = cv2.VideoCapture(video_patch)
  while(vidcap.isOpened()):

    # Capture frame-by-frame
    ret, frame = vidcap.read()
    if ret:
      cv2.imshow('video', frame)
      assert not isinstance(frame,type(None)), 'frame not found'
      if cv2.waitKey(10) == 27:  # Esc to exit
        break
    else: break
  vidcap.release()
  cv2.destroyAllWindows()
  

#2 Redimensionar el frame
def resize_frame(frame, neural_net, input_blob):
  B, C, H, W = neural_net.input_info[input_blob].tensor_desc.dims
  resized_frame = cv2.resize(frame, (W, H))
  initial_h, initial_w, _ = frame.shape


# #3 Recortar el frame con Opencv
# #vidcap = cv2.VideoCapture(video_patch)
# imagen = cv2.imread('pictures/picture_1.png')
# alto, ancho, canales = imagen.shape
# print('Alto={}, Ancho={}, Canales={}'.format(alto, ancho, canales))


#FPS Counter
def fps_counter():
    dt = datetime.now()
    ts = int(datetime.timestamp(dt))
    if ts > initial_ts:
        print("FPS: ", fps)
        fps = 0
        initial_ts = ts
    else:
        fps += 1


# Funcion para graficar resultados sobre el frame
def drawText(frame, scale, rectX, rectY, rectColor, text):
    rectThinkness = 2
    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    top = max(rectY - rectThinkness, textSize[0])
    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )