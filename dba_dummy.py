import imutils, cv2, json
from openvino.inference_engine import IECore
from face_detection.functions import fps_counter, generate_roi, face_detection
from face_reidentification.functions import process, face_comparison, face_recognition

with open("face_detection/settings.json") as settings:
    config = json.load(settings)

# OpenVino models
model_xml = config.get("model_xml")
model_bin = config.get("model_bin")

# Video location
video_patch = config.get("video_patch")

# Device
device = config.get("device")

def main():

    ie = IECore() # Instantiate an IEcore object to work with openvino

    neural_net = ie.read_network(
        model_xml, 
        model_bin
    )
    execution_net = ie.load_network(
        network=neural_net, device_name=device.upper()
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
        ret, frame = vidcap.read() # Capture frame-by-frame
        if ret:
            frame = frame[cropped_frame[0][1] : cropped_frame[1][1],cropped_frame[0][0] : cropped_frame[1][0]]
            face_detection(frame, neural_net, execution_net, input_blob, output_blob, detection_area)
            face_recognition(frame)
            face_comparison(new_vector=0)
            process(frame, ("faces"))
            if cv2.waitKey(15) == 27:  # Esc to exit
                break
        else: break

        fps_counter(frame)

        showImg = imutils.resize(frame, height=500)
        cv2.imshow('Live Streaming', showImg) # Display frame/image

    vidcap.release() # Release video capture object
    cv2.destroyAllWindows() # Destroy all frame windows

main()
print("------------------------------")
print("USE CASE EXECUTED SUCCESSFULLY")
print("------------------------------")