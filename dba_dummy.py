from datetime import datetime
import imutils
import cv2
import numpy as np
from openvino.inference_engine import IECore
import json
from fpsCounter import fps_counter

with open("settings.json") as settings:
    config = json.load(settings)

# OpenVino models
model_xml = config.get("model_xml")
model_bin = config.get("model_bin")

# Video location
video_patch = config.get("video_patch")

# Colors to be used with opencv
BLUE = config.get("BLUE")
RED = config.get("RED")

# Parameter to filter detections based on confidence
confidence = config.get("confidence")

# Device
device = config.get("device")

# Select area of interest function
def generate_roi(frame, message):
    # Obtain the image height and width
    frame_height, frame_width = frame.shape[:-1]
    # Determines a crop area
    detection_area = [[0, 0], [frame_width, frame_height]]
    top_left_crop = (0, 0)
    bottom_right_crop = (frame_width, frame_height)
    # Select detection area
    window_name_roi = message # Window Name
    roi = cv2.selectROI(window_name_roi, frame, False)
    cv2.destroyAllWindows() # Destroys all the windows we created
    #organize the results into a list of 2 tuples to be processed by check_detection area
    if int(roi[2]) != 0 and int(roi[3]) != 0:
        x_tl, y_tl = int(roi[0]), int(roi[1])
        x_br, y_br = int(roi[0] + roi[2]), int(roi[1] + roi[3])
        detection_area = [
            (x_tl, y_tl),
            (x_br, y_br),
        ]
    else: # If nothing is selected it takes the whole frame
        detection_area = [
            (0, 0),
            (
                bottom_right_crop[0] - top_left_crop[0],
                bottom_right_crop[1] - top_left_crop[1],
            ),
        ]
    return detection_area # Returns the values

# Check detection area
def check_detection_area(x, y, detection_area):
    # Verify that the detection area has size 2 if it does not throw an error
    if len(detection_area) != 2: 
        raise ValueError("Invalid number of points in detection area")
    # Set boundaries area
    top_left = detection_area[0]
    bottom_right = detection_area[1]
    # Get coordinates
    xmin, ymin = top_left[0], top_left[1]
    xmax, ymax = bottom_right[0], bottom_right[1]
    # Returns True if the passed parameters are within the detection area
    return xmin < x and x < xmax and ymin < y and y < ymax

# Main Function
def face_detection( # Get parameters of the model
    frame,
    neural_net,
    execution_net,
    input_blob,
    output_blob,
    detection_area
):

    # B: batch size, C: number of channels, H: image height, W: image width
    B, C, H, W = neural_net.input_info[input_blob].tensor_desc.dims
    # Resizes the frame according to the parameters of the model
    resized_frame = cv2.resize(frame, (W, H)) # Resize the frame
    initial_h, initial_w, _ = frame.shape # Sets height and width based on the dimensions of the array
    # Format the array to have the shape specified in the model
    # Let element 3 first, 1 second, and 2 third and then adding a new one in position 1
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
    results = execution_net.infer(inputs={input_blob: input_image}).get(output_blob)

    for detection in results[0][0]:
        label = int(detection[1])
        accuracy = float(detection[2])
        det_color = BLUE if label == 1 else RED
        # Draw only objects when accuracy is greater than configured threshold
        if accuracy > confidence:
            xmin = int(detection[3] * initial_w)
            ymin = int(detection[4] * initial_h)
            xmax = int(detection[5] * initial_w)
            ymax = int(detection[6] * initial_h)
            # Central points of detection
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2

            # Check if central points fall inside the detection area
            if check_detection_area(x, y, detection_area):
                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    det_color,
                    thickness=2,
                )

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