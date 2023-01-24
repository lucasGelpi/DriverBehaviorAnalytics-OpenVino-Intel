import cv2, json
import numpy as np
from imutils import paths
from scipy import spatial
import os
import logging
from openvino.inference_engine import IECore

with open("face_reidentification/settings.json") as settings:
    config = json.load(settings)

# # OpenVino models
# model_xml = config.get("model_xml")
# model_bin = config.get("model_bin")

# # Device
# device = config.get("device")

# # Parameter to filter detections based on confidence
# confidence = config.get("confidence")

# drivers_dir = config.get("drivers_dir")
# drivers_dict = {}
# driver_name = "Unknown"

class Udf:
    """Address Detection UDF"""

    def __init__(self, model_xml, model_bin, device, confidence_threshold, drivers_dir):
        """Constructor"""
        self.log = logging.getLogger("FACE_REIDENTIFICATION")
        self.model_xml = config.get("model_xml")
        self.model_bin = config.get("model_bin")
        self.device = config.get("device")
        self.confidence_threshold = float(confidence_threshold)
        self.drivers_dir = config.get("drivers_dir")
        self.drivers_dict = {}
        self.new_driver = True
        self.driver_name = "Unknown"

        if not os.path.exists(self.model_xml):
            raise FileNotFoundError(f"Model xml file missing: {self.model_xml}")
        if not os.path.exists(self.model_bin):
            raise FileNotFoundError(f"Model bin file missing: {self.model_bin}")
        self.log.info("Config reading completed...")
        self.log.info("Confidence = %s", self.confidence_threshold)
        self.log.info(
            "Loading IR files. \n\txml: %s, \n\tbin: %s", self.model_xml, self.model_bin
        )

        # Load OpenVINO model
        self.ie_core = IECore()
        self.neural_net = self.ie_core.read_network(model=model_xml, weights=model_bin)
        if self.neural_net:
            self.input_blob = next(iter(self.neural_net.input_info))
            self.neural_net.batch_size = 1
            self.execution_net = self.ie_core.load_network(
                network=self.neural_net, device_name=device.upper()
            )
            self.output_blob = next(iter(self.execution_net.outputs))

        for image_path in paths.list_images(self.drivers_dir):
            name = os.path.basename(image_path).split(".")[0]
            if name.startswith("driver_"):
                name = name[len("driver_"):]
            name = name.replace("_", " ")
            try:
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except (IOError, cv2.error):
                img = None
                self.log.warning(f"Invalid file: {image_path}")
            if img is not None:
                self.drivers_dict[name] = self.face_recognition(img)

        self.log.debug(f"DRIVERS: {self.drivers_dict}")

    def face_recognition(self, frame):

        _, _, H, W = self.neural_net.input_info[self.input_blob].tensor_desc.dims
        resized_frame = cv2.resize(frame, (W, H))

        # reshape to network input shape
        # Change data layout from HWC to CHW
        input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

        face_recognition_results = self.execution_net.infer(
            inputs={self.input_blob: input_image}
        ).get(self.output_blob)

        return [x[0][0] for x in list(face_recognition_results[0])]

    def face_comparison(self, new_vector):
        for name, vector in self.drivers_dict.items():
            result = 1 - spatial.distance.cosine(vector, new_vector)
            if result >= self.confidence_threshold:
                return name
        return "Unknown"

    def process(self, frame, metadata):
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
        if faces and self.drivers_dict:
            if self.new_driver:
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
                    vector = self.face_recognition(frame)
                    self.driver_name = self.face_comparison(vector)
                    self.new_driver = self.driver_name == "Unknown"

        else:
            self.new_driver = True
            self.driver_name = "Unknown"
            self.log.debug("No driver detected.")

        metadata["driver_name"] = self.driver_name.encode(
            'ASCII', 'surrogateescape').decode('UTF-8')
        return False, None, metadata