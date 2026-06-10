import cv2
from deepface import DeepFace

interval = 5
batch_size = 8
# List[pd.DataFrame] = DeepFace.find(img_path = "img1.jpg", db_path = "C:/my_db")
"""
DEEP FACE DETECTION:
    - faceDetection: returns bbox for every face in faces (x, y, w, h) + emotion label
    - annotateEmotion: returns emotion label for given cropped face image
"""


class DetectFace:
    def __init__(self):
        # self.frame = frame
        pass

    def annotateEmotion(self, imagecrop):
        try:
            emo_result = DeepFace.analyze(
                imagecrop, actions=["emotion"], enforce_detection=False
            )
            # dela le za prvi zaznan face, 0 ker vzamemo prvi face ven?
            detected_emotion = emo_result[0]["dominant_emotion"]
            return detected_emotion
        except Exception as e:
            print(f"Napaka pri detekciji emocije: {e}")
            return None

    def faceDetection(self, frame):
        try:
            faces = DeepFace.extract_faces(
                frame,
                detector_backend="mediapipe",
                enforce_detection=False,
                # zakaj ssd da conf 0?? opencv pa dela?
                # moram pravilno resizeat image za ssd
            )
            # print("Detected faces keys:", faces[0].keys() if faces else "None")
            multiple_bboxes = []
            # if faces is not None:
            #     for detected_face in faces:
            for detected_face in faces if faces is not None else []:
                # Uporabimo MOSSE tracker za hitrost, za vsak detected face ki dobimo iz "faces"
                # tracker = cv2.TrackerMOSSE_create()
                # fa koordinate obraza za bbox shranimo koordinate za trenutni face
                fa = detected_face["facial_area"]
                x, y, w, h = (
                    fa["x"],
                    fa["y"],
                    fa["w"],
                    fa["h"],
                )
                print(
                    f"Detected DEEPFACE at ({x}, {y}, {w}, {h}) with confidence: {detected_face.get('confidence', 'N/A')}"
                )
                # tracker.init(frame, (x, y, w, h))
                # trackers.append(tracker)

                cropped_face = frame[y : y + h, x : x + w]
                emotion = self.annotateEmotion(cropped_face)
                if emotion is not None:
                    print(
                        f"Detected emotion for face at ({x}, {y}, {w}, {h}): {emotion}"
                    )

                multiple_bboxes.append(
                    {
                        "bbox": (x, y, w, h),
                        "conf": detected_face.get("confidence", None),
                        "emo": emotion,
                    }
                )
                ##### tale dela sama po seb za en obraz #####
                # fa = faces[0]["facial_area"]
                # # print(f"Found facial area: {fa}")
                # x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                # return x, y, w, h
            return multiple_bboxes
        except Exception as e:
            print(f"Napaka pri detekciji obraza: {e}")
            return None


if __name__ == "__main__":
    DetectFace()

    ###############
    # detections = det_class.faceDetection(frame)
    # print("confidence:", detections.get("confidence"))
    # print(
    #     "left_eye:",
    #     fa.get("left_eye"),
    #     "right_eye:",
    #     fa.get("right_eye"),
    # )
    ##################
