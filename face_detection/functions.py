from datetime import datetime
import cv2
import numpy as np
import json

with open("face_detection/settings.json") as settings:
    config = json.load(settings)

# Colors to be used with opencv
BLUE = config.get("BLUE")
RED = config.get("RED")

# Parameter to filter detections based on confidence
confidence = config.get("confidence")

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
    metadata = {}
    metadata["faces"] = []
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
                metadata["faces"].append({
                    'tl': [xmin, ymin],
                    'br': [xmax, ymax]
                })
    return metadata

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

# Set variables to calculate fps
initial_dt = datetime.now()
initial_ts = int(datetime.timestamp(initial_dt))

# Set the frames to 0 before starting to count
fps = config.get("fps")
save_fps = config.get("save_fps")

# FPS Counter
def fps_counter(frame):
    global initial_dt, initial_ts, fps, save_fps # Make global variables
    dt = datetime.now()
    ts = int(datetime.timestamp(dt))
    if ts > initial_ts:
        save_fps = fps # Save results in a variable
        fps = 0 # Set fps to 0
        initial_ts = ts
    else:
        fps += 1
    font = cv2.FONT_HERSHEY_SIMPLEX # Font which we will be using to display FPS
    cv2.putText(frame, "FPS: " + str(int(save_fps)), (5, 50), font, 1, (0, 255, 255), 2) #Print FPS on the frame