import cv2, json
import numpy as np
from imutils import paths
from scipy import spatial
import os
import logging as log
from openvino.inference_engine import IECore

with open("face_reidentification/settings.json") as settings:
    config = json.load(settings)

# OpenVino models
model_xml = config.get("model_xml")
model_bin = config.get("model_bin")

# Device
device = config.get("device")

# Parameter to filter detections based on confidence
confidence = config.get("confidence")

drivers_dir = config.get("drivers_dir")
drivers_dict = {}
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

log.debug(f"DRIVERS: {drivers_dict}")

def face_recognition(frame):

    _, _, H, W = neural_net.input_info[input_blob].tensor_desc.dims
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

for image_path in paths.list_images(drivers_dir):
    name = os.path.basename(image_path).split(".")[0]
    if name.startswith("driver_"):
        name = name[len("driver_"):]
    name = name.replace("_", " ")
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (IOError, cv2.error):
        img = None
        log.warning(f"Invalid file: {image_path}")
    if img is not None:
        drivers_dict[name] = face_recognition(img)

def face_comparison(new_vector):
    for name, vector in drivers_dict.items():
        result = 1 - spatial.distance.cosine(vector, new_vector)
        if result >= confidence:
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