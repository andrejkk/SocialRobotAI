from deepface import DeepFace
import cv2

import os
from pathlib import Path
import matplotlib.pyplot as plt

interval = 5
batch_size = 8
# List[pd.DataFrame] = DeepFace.find(img_path = "img1.jpg", db_path = "C:/my_db")
"""DEEP FACE MODULE"""


class DetectFace:
    def __init__(self):
        # self.frame = frame
        pass

    # print("Pri deepface smo prisli not")
    trackers = []

    def annotateEmotion(self, imagecrop):
        try:
            emo_result = DeepFace.analyze(
                imagecrop, actions=["emotion"], enforce_detection=False
            )
            # dominant_emotion = emo_result.get("emotion").get("dominant_emotion") get dela samo na seznamih!
            # jaz imam dict čeprav je hierarhija dicta shranjena v seznamu
            # dom_emo = emo_result["emotion"][0]
            # dom_emo = emo_result["dominant_emotion"]
            print(f"Detected dominant emotion: {emo_result}")
            detected_emotions = emo_result[0]
            dom_emo = detected_emotions["dominant_emotion"]
            # print("Detected emotions: ", emo_result, "are type of: ", type(emo_result))
            # print("Detected emotions:", emo_result.get("emotion"))
            # print("Dominant emotion:", emo_result.get("dominant_emotion"))
            return dom_emo
        except Exception as e:
            print(f"Napaka pri detekciji emocije: {e}")
            return None

    def faceDetection(self, frame):
        # out_dir = Path("cropped_faces")
        # if not out_dir.exists():
        #     out_dir.mkdir(parents=True, exist_ok=True)
        # else:
        #     pass

        try:
            faces = DeepFace.extract_faces(frame, detector_backend="opencv")
            print("Detected faces keys:", faces[0].keys() if faces else "None")
            multiple_bboxes = []
            if faces is not None:
                # for f in faces:
                #     fa = f["facial_area"]
                #     print(f"Found facial area: {fa}")
                #     x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                #     # return x, y, w, h
                i = 0
                for detected_face in faces:
                    # dostopaj do dobljenih koordinat iz detektiranga obraza izmed vseh detektiranih
                    facial_area = detected_face["facial_area"]

                    # print("Detected face keys:", detected_face.keys())

                    iidface_cropped = detected_face["face"]
                    # print("A: Detected emotions:", emo_result.get("emotion"))
                    # print("B: Dominant emotion:", emo_result.get("dominant_emotion"))
                    # detected_emotion = detected_face["dominant_emotion"]
                    # print(f"detected emotion: {detected_emotion}")

                    # shranimo koordinate za trenutni face kje je bil detektiran obraz (ocrtan okvir)
                    x, y, w, h = (
                        facial_area["x"],
                        facial_area["y"],
                        facial_area["w"],
                        facial_area["h"],
                    )
                    cropped_face = frame[y : y + h, x : x + w]
                    detected_emotion = self.annotateEmotion(cropped_face)
                    if detected_emotion is not None:
                        print(
                            f"Detected emotion for face at ({x}, {y}, {w}, {h}): {detected_emotion}"
                        )
                        # cv2.putText()

                    # if i == 0:
                    #     cropped_face = frame[y : y + h, x : x + w]
                    #     plt.figure(figsize=(6, 4))
                    #     plt.plot(cropped_face)

                    #     filename = out_dir / f"frame_{frame:03d}.png"
                    #     plt.savefig(filename)
                    #     plt.imshow(cropped_face)
                    #     plt.close()  # important to free memory
                    #     print(f"Saved {filename}")
                    #     i = 1
                    # else:
                    #     pass

                    multiple_bboxes.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                            detected_emotion,
                        )
                    )
                    # returnam tudi none vrednost? kaj pa če imam tu if?
                return multiple_bboxes

                ##### tale dela sama po seb za en obraz #####
                # fa = faces[0]["facial_area"]
                # # print(f"Found facial area: {fa}")
                # x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                # return x, y, w, h
        except Exception as e:
            print(f"Napaka pri detekciji obraza: {e}")
            return None


if __name__ == "__main__":
    DetectFace()
    # cap = cv2.VideoCapture(0)

    # detected_faces = []

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Napaka pri zajemanju videa")
    #         break

    ###############
    # detections = det_class.faceDetection(frame)
    # print(f"Detections: {detections}")
    # if detections is not None:
    # if attr == "facial_area":
    # fa = detections.get("facial_area")

    # TODO: iz detections vzemi vse zaznane obraze, ne samo prvega [0]
    # for face in detections:
    #     fa = face["facial_area"]
    #     detected_faces.append(fa)

    # print(len(detected_faces))

    # fa = detections[0]["facial_area"]

    # print(f"Found facial area: {fa}")
    # x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
    # # draw rectanglecv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print("confidence:", detections.get("confidence"))
    # print(
    #     "left_eye:",
    #     fa.get("left_eye"),
    #     "right_eye:",
    #     fa.get("right_eye"),
    # )
    #########################################
    # if detections is not None:
    #     for attr in detections:
    #         if attr == "facial_area":
    #             # fa = detections.get("facial_area")
    #             fa = detections[0]["facial_area"]

    #             print(f"Found facial area: {fa}")
    #             x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
    #             # draw rectanglecv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             print("confidence:", detections.get("confidence"))
    #             print(
    #                 "left_eye:",
    #                 fa.get("left_eye"),
    #                 "right_eye:",
    #                 fa.get("right_eye"),
    #             )

    #     extr_face = detections["facial area"]
    #     x, y, w, h = extr_face["x"], extr_face["y"], extr_face["w"], extr_face["h"]

    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     # for x, y, w, h in detections:
    #     #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow("Face Detection", frame)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
