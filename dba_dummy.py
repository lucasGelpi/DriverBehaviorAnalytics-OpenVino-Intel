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
device = "CPU"

ie = IECore()

#2 Redimensionar el frame
def resize_frame(frame, neural_net, input_blob):
    neural_net = ie.read_network(
    model=model_xml, weights=model_bin)
    B, C, H, W = neural_net.input_info[input_blob].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))
    initial_h, initial_w, _ = frame.shape

#3 Recortar el frame con Opencv
def recortar_imagen(frame):
    img = frame.copy()
    h, w, c = img.shape
    imgC1 = img[10:350, 10:590]
    # video = cv2.imshow('imgC1', imgC1)
    return imgC1

# Funcion para graficar resultados sobre el frame
def drawText(frame, scale, rectX, rectY, rectColor, text):
    rectThinkness = 2
    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    top = max(rectY - rectThinkness, textSize[0])
    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )

def main():

    ie = IECore()

    neural_net = ie.read_network(
        model=model_xml, weights=model_bin
    )
    car_pedestrian_execution_net = ie.load_network(
        network=neural_net, device_name=device.upper()
    )
    input_blob = next(iter(car_pedestrian_execution_net.input_info))
    output_blob = next(iter(car_pedestrian_execution_net.outputs))
    neural_net.batch_size = 1

    initial_dt = datetime.now()
    initial_ts = int(datetime.timestamp(initial_dt))
    fps = 0

    #1 Obtener el frame
    vidcap = cv2.VideoCapture(video_patch)
    while(vidcap.isOpened()):

        # Capture frame-by-frame
        ret, frame = vidcap.read()
        
        if ret:
            resize_frame(frame, neural_net, input_blob)
            recortar_imagen(frame)

            assert not isinstance(frame,type(None)), 'frame not found'
            if cv2.waitKey(10) == 27:  # Esc to exit
                break
        else: break

        #FPS Counter
        dt = datetime.now()
        ts = int(datetime.timestamp(dt))

        if ts > initial_ts:
            print("FPS: ", fps) # Print FPS in console
            fps = 0
            initial_ts = ts
        else:
            fps += 1
        
        fps = int(vidcap.get(cv2.CAP_PROP_FPS)) # Acces FPS property

        font = cv2.FONT_HERSHEY_SIMPLEX # Font to apply on text
        cv2.putText(frame, str(fps), (50,50), font, 1, (0, 0, 255), 2) # Add text on frame
        cv2.imshow('Live Streaming', frame) # Display frame/image

    vidcap.release() # Release video capture object
    cv2.destroyAllWindows() # Destroy all frame windows

if __name__ == "__main__":
    main()