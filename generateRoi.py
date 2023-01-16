import cv2

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