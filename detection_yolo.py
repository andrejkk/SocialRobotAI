import cv2
import numpy as np
from ultralytics import YOLO

###deepface
# import faceDetection from deepface

# import torch

"""
#### Detekcija ground truth z YOLO, nato deepface zaznava obraza ####
"""

"""
TODO:
- Nastaviti device za CUDA core:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
- Posredovati conf_thresh pri klicu detektorja v args (kako ga določimo?)
- Optimizirati detekcijo in frame rate za delovanje v realtime (threading?)
"""


class DetectorPerson:
    # konstruktor YOLO objekta za ground truth detection
    def __init__(self, model_name="yolov8m.pt", confidence_threshold=0.5):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    # funkcija za pridobiti bounding box/ground truth iz trenutnega okvirja, anotacija
    def DetectBoundingBox(self, currentFrame):
        detection_results = self.model(currentFrame, conf=self.confidence_threshold)
        detections_array = []
        print("problem")
        for result in detection_results:
            for bounding_box in result.boxes:
                if bounding_box.conf > self.confidence_threshold:
                    # bounding_box.append(bounding_box.xyxy.cpu().numpy())
                    x1, y1, x2, y2 = map(int, bounding_box.xyxy[0])
                    detections_array.append(
                        {
                            "box": (x1, y1, x2, y2),
                            "confidence": float(bounding_box.conf),
                        }
                    )

                    cv2.rectangle(currentFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        currentFrame,
                        f"Person {float(bounding_box.conf):.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
        return currentFrame, detections_array

    """
    TODO:
        - importati video iz data in preprocesirati format
    """

    # funkcija za procesirat video okvirje
    def ProcessStream(self, stream_source_arg):
        capture = cv2.VideoCapture(stream_source_arg)
        # while not capture:
        #     box, frame = capture.read()
        #     if box:
        #         bounding_box = self.DetectBoundingBox(frame)

        #     else:
        #         print("Napaka pri procesiranju videa")
        #         break
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Napaka pri zajemanju videa")
                break

            frame, detections = self.DetectBoundingBox(frame)
            print(f"Detected {len(detections)} people")

            cv2.imshow("YOLO People Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorPerson(
        model_name="yolov8m.pt", confidence_threshold=0.5  # spreminjaj model po potrebi
    )
    """
    pošlji v detektor webcam z args 0 od opencv za testiranje
    """
    # webcam = cv2.VideoCapture(0)
    detector.ProcessStream(0)
