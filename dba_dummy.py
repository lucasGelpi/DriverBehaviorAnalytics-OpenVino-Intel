import time
import imutils
import cv2
import numpy as np
from openvino.inference_engine import IECore

model_bin = "./models/face-detection-retail-0005.bin"
model_xml = "./models/face-detection-retail-0005.xml"
video_patch = "./video/Face_Cam_1.mp4"
BLUE = (255, 0, 0)
RED = (0, 0, 255)
confidence_threshold = 0.6
device = "CPU"

# Function to select area of interest
def generate_detection_area(frame):
    # By default, keep the original frame and select complete area
    frame_height, frame_width = frame.shape[:-1]
    detection_area = [[0, 0], [frame_width, frame_height]]
    top_left_crop = (0, 0)
    bottom_right_crop = (frame_width, frame_height)
    # Select detection area
    window_name_roi = "Select Detection Area."
    roi = cv2.selectROI(window_name_roi, frame, False)
    cv2.destroyAllWindows()
    if int(roi[2]) != 0 and int(roi[3]) != 0:
        x_tl, y_tl = int(roi[0]), int(roi[1])
        x_br, y_br = int(roi[0] + roi[2]), int(roi[1] + roi[3])
        detection_area = [
            (x_tl, y_tl),
            (x_br, y_br),
        ]
    else:
        detection_area = [
            (0, 0),
            (
                bottom_right_crop[0] - top_left_crop[0],
                bottom_right_crop[1] - top_left_crop[1],  ## Revisar
            ),
        ]
    return detection_area

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
    # Check if the point is inside a ROI
    return xmin < x and x < xmax and ymin < y and y < ymax

# Main Function
def face_detection( #obtiene parametros del modelo
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
        if accuracy > confidence_threshold:
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
        model = model_xml, 
        weights = model_bin
    )
    execution_net = ie.load_network(
        network=neural_net, device_name=device.upper()
    )
    input_blob = next(iter(execution_net.input_info))
    output_blob = next(iter(execution_net.outputs))
    neural_net.batch_size = 1 # Number of frames processed in parallel

    vidcap = cv2.VideoCapture(video_patch) #1 Capture the frame using a video as source
    success, frame = vidcap.read() # Capture frame-by-frame

    # Crop the frame by setting the detection area
    detection_area = generate_detection_area(frame)

    fps_start_time = 0
    fps = 0

    while(success): # Reading the video file until finished
        ret, frame = vidcap.read() # Capture frame-by-frame

        if ret:
            face_detection(frame, neural_net, execution_net, input_blob, output_blob, detection_area)
        else: break
        
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1/(time_diff)
        fps_start_time = fps_end_time
        fps_text = "FPS: {:.0f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        showImg = imutils.resize(frame, height=600)
        cv2.imshow('Live Streaming', showImg) # Display frame/image
        
        if cv2.waitKey(10) == 27:  # Esc to exit
            break

    vidcap.release() # Release video capture object
    cv2.destroyAllWindows() # Destroy all frame windows

main()
print("------------------------------")
print("USE CASE EXECUTED SUCCESSFULLY")
print("------------------------------")