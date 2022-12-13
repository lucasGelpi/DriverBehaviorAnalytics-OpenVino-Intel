import cv2
from openvino.inference_engine import IECore

model_bin = "./models/face-detection-retail-0005.bin"
model_xml = "./models/face-detection-retail-0005.xml"
video_Patch = "./video/Driver_2_Face_Cam.mp4"

# def generate_detection_area(frame):
#     # By default, keep the original frame and select complete area
#     frame_height, frame_width = frame.shape[:-1]
#     detection_area = [[0, 0], [frame_width, frame_height]]
#     top_left_crop = (0, 0)
#     bottom_right_crop = (frame_width, frame_height)
#     # Select detection area
#     window_name_roi = "Select Detection Area."
#     roi = cv2.selectROI(window_name_roi, frame, False)
#     cv2.destroyAllWindows()
#     if int(roi[2]) != 0 and int(roi[3]) != 0:
#         x_tl, y_tl = int(roi[0]), int(roi[1])
#         x_br, y_br = int(roi[0] + roi[2]), int(roi[1] + roi[3])
#         detection_area = [
#             (x_tl, y_tl),
#             (x_br, y_br),
#         ]
#     else:
#         detection_area = [
#             (0, 0),
#             (
#                 bottom_right_crop[0] - top_left_crop[0],
#                 bottom_right_crop[1] - top_left_crop[1],
#             ),
#         ]
#     return detection_area

# print(generate_detection_area)

vid_capture = cv2.VideoCapture(video_Patch)
while (vid_capture.isOpened()):
  ret, img = vid_capture.read()
  if ret == True:
    cv2.imshow('video', img)
    if cv2.waitKey(30) == 27:  # exit if Escape is hit
        break
  else: break
vid_capture.release()
cv2.destroyAllWindows()

def face_Detection(
    frame,
    neural_net,
    execution_net,
    input,
    output,
    detection_area,
):
  N, C, H, W = neural_net.input_info
  return frame

