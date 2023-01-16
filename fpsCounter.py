from datetime import datetime
import cv2
import json

with open("settings.json") as settings:
    config = json.load(settings)

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
    cv2.putText(frame, "FPS:" + str(int(save_fps)), (5, 30), font, 1, (0, 255, 255), 2) #Print FPS on the frame