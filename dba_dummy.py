import logging
import os
import imutils, cv2, json
from openvino.inference_engine import IECore
from face_detection.functions import fps_counter, generate_roi, face_detection
# from face_reidentification.functions import Udf
from imutils import paths
import numpy as np
from scipy import spatial

############################################################################

with open("face_reidentification/settings.json") as settings:
    config = json.load(settings)

log = logging.getLogger("FACE_REIDENTIFICATION")
model_xml = config.get("model_xml")
model_bin = config.get("model_bin")
device = config.get("device")
confidence = config.get("confidence ")
drivers_dir = config.get("drivers_dir")
drivers_dict = {}
new_driver = True
driver_name = "Unknown"

if not os.path.exists(model_xml):
    raise FileNotFoundError(f"Model xml file missing: {model_xml}")
if not os.path.exists(model_bin):
    raise FileNotFoundError(f"Model bin file missing: {model_bin}")
log.info("Config reading completed...")
log.info("Confidence = %s", confidence)
log.info(
    "Loading IR files. \n\txml: %s, \n\tbin: %s", model_xml, model_bin
)

def face_recognition(frame):

    B, C, H, W = neural_net.input_info[input_blob].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))

    # reshape to network input shape
    # Change data layout from HWC to CHW
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    face_recognition_results = execution_net.infer(
        inputs={input_blob: input_image}
    ).get(output_blob)

    return [x[0][0] for x in list(face_recognition_results[0])]

# Load OpenVINO model
ie_core = IECore()
neural_net = ie_core.read_network(model=model_xml, weights=model_bin)
if neural_net:
    input_blob = next(iter(neural_net.input_info))
    neural_net.batch_size = 1
    execution_net = ie_core.load_network(
        network=neural_net, device_name=device.upper()
    )
    output_blob = next(iter(execution_net.outputs))

video_patch = config.get("video_patch")
vidcap = cv2.VideoCapture(video_patch) #1 Capture the frame using a video as source
success, frame = vidcap.read()

for image_path in paths.list_images(drivers_dir):
    name = os.path.basename(image_path).split(".")[0]
    if name.startswith("driver_"):
        name = name[len("driver_"):]
    name = name.replace("_", " ")
    try:
        frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (IOError, cv2.error):
        frame = None
        print()
        log.warning(f"Invalid file: {image_path}")
    if frame is not None:
        drivers_dict[name] = face_recognition(frame)

log.debug(f"DRIVERS: {drivers_dict}")

def face_comparison(self, new_vector):
    for name, vector in self.drivers_dict.items():
        result = 1 - spatial.distance.cosine(vector, new_vector)
        if result >= self.confidence_threshold:
            return name
    return "Unknown"

def process(frame, metadata):
    """[summary]
    :param frame: frame blob
    :type frame: numpy.ndarray
    :param metadata: frame's metadata
    :type metadata: str
    :return:  (should the frame be dropped, has the frame been updated,
                new metadata for the frame if any)
    :rtype: (bool, numpy.ndarray, str)
    """
    faces = metadata.get("faces")
    if faces and drivers_dict:
        if new_driver:
            face = faces[0]
            xmin, ymin = face["tl"]
            xmax, ymax = face["br"]
            frame = frame[ymin : ymax + 1, xmin : xmax + 1]
            if frame.any():
                frame = cv2.resize(
                    frame,
                    (
                        xmax - xmin,
                        ymax - ymin,
                    ),
                )
                vector = face_recognition(frame)
                driver_name = face_comparison(vector)
                new_driver = driver_name == "Unknown"

    else:
        new_driver = True
        driver_name = "Unknown"
        log.debug("No driver detected.")

    metadata["driver_name"] = driver_name.encode(
        'ASCII', 'surrogateescape').decode('UTF-8')
    return False, None, metadata

#########################################################################

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
            
            if cv2.waitKey(15) == 27:  # Esc to exit
                break
        else: break

        fps_counter(frame)

        face_recognition(frame)

        process(frame, meta)
        
        showImg = imutils.resize(frame, height=500)
        cv2.imshow('Live Streaming', showImg) # Display frame/image

    vidcap.release() # Release video capture object
    cv2.destroyAllWindows() # Destroy all frame windows

main()
print("------------------------------")
print("USE CASE EXECUTED SUCCESSFULLY")
print("------------------------------")