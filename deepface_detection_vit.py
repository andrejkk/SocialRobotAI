import torch
import cv2
from PIL import Image
from transformers import (
    Owlv2Processor,
    Owlv2ForObjectDetection,
)

class DetectFace:
    def __init__(self, detector="owlv2", DeepFace=None):

        self.detector = detector
        self.DeepFace = DeepFace

        if detector == "owlv2":

            self.processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )

            self.model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )

            self.model.eval()
            self.text_labels = [["photo of a human face"]] #a se da se bolj natancno nastavit text??
    
    def faceDetection(self, frame):
        if frame is None:
            return None

        try: 
            if self.detector == "owlv2":
                return self.owlv2_detection(frame)

            elif self.detector == "deepface":
                return self.deepface_detection(frame)
        except Exception as e:
            #print(f"Face detection error: {e}", flush=True)
            return None
        
    def owlv2_detection(self, frame):
        # dam v BGR -> RGB, razn ce to ze nrdim ze prej v cevovodu
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        inputs = self.processor(
            text=self.text_labels,
            images=image,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        target_sizes = [
            (image.height, image.width)
        ]
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=0.10, #optimalen threshlold??
            target_sizes=target_sizes,
            text_labels=self.text_labels
        )

        result = results[0]

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"]

        detections = []

        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):

            x1, y1, x2, y2 = box.tolist()

            detections.append({
                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1)
                ),
                "conf": float(score),
                "emo": None
            })
       #print("THE DETECTIONS I GOT: ", detections)
        return detections


def process_handler(request_queue, result_queue, stop_event, ready_event):
    df = DetectFace(detector="owlv2")
    ready_event.set()
    while not stop_event.is_set():
        frameid, frame = request_queue.get()
        det_result = df.faceDetection(frame)
        result_queue.put((frameid, det_result))