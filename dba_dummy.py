import imutils, cv2, json
from openvino.inference_engine import IECore
from face_detection.functions import fps_counter, generate_roi, face_detection
from imutils import paths
import numpy as np
from scipy import spatial
from face_reidentification.functions import FaceReidClass

with open("face_detection/settings.json") as settings:
    configDet = json.load(settings)

with open("face_reidentification/settings.json") as settings:
    configRei = json.load(settings)

# OpenVino models
model_xmlDet = configDet.get("model_xml")
model_binDet = configDet.get("model_bin")
model_xmlRei = configRei.get("model_xml")
model_binRei = configRei.get("model_bin")

# Video location
video_patch = configDet.get("video_patch")

# Device
deviceDet = configDet.get("device")

confidenceRei = configRei.get("confidence")
deviceRei = configRei.get("device")
drivers_dir = configRei.get("drivers_dir")

reidClas = FaceReidClass(model_xmlRei, model_binRei, deviceRei, confidenceRei, drivers_dir)

def main():

    ie = IECore() # Instantiate an IEcore object to work with openvino

    neural_net = ie.read_network(
        model_xmlDet, 
        model_binDet
    )
    execution_net = ie.load_network(
        network=neural_net, device_name=deviceDet.upper()
    )
    input_blob = next(iter(execution_net.input_info))
    output_blob = next(iter(execution_net.outputs))
    neural_net.batch_size = 1 # Number of frames processed in parallel

    vidcap = cv2.VideoCapture(video_patch) #1 Capture the frame using a video as source
    
    # Returns a tuple with a boolean and the data of the frame in the form of an array
    success, frame = vidcap.read()

    # Crop the frame setting the detection area
    cropped_frame = generate_roi(frame, "Select Crop Area")

    frame = frame[cropped_frame[0][1] : cropped_frame[1][1],cropped_frame[0][0] : cropped_frame[1][0]]
    frame = cv2.resize(frame,(cropped_frame[1][0] - cropped_frame[0][0],cropped_frame[1][1] - cropped_frame[0][1]))

    # Crop the frame by setting the detection area
    detection_area = generate_roi(frame, "Select Detection Area")

    while(success): # Reading the video file until finished
        metadata = {}
        ret, frame = vidcap.read() # Capture frame-by-frame
        if ret:
            frame = frame[cropped_frame[0][1] : cropped_frame[1][1],cropped_frame[0][0] : cropped_frame[1][0]]
            metadata = face_detection(frame, neural_net, execution_net, input_blob, output_blob, detection_area)

            if cv2.waitKey(10) == 27:  # Esc to exit
                break
        else: break

        fps_counter(frame)

        reidClas.process(frame, metadata)
        
        showImg = imutils.resize(frame, height=500)
        cv2.imshow('Live Streaming', showImg) # Display frame/image

    vidcap.release() # Release video capture object
    cv2.destroyAllWindows() # Destroy all frame windows

main()
print("------------------------------")
print("USE CASE EXECUTED SUCCESSFULLY")
print("------------------------------")