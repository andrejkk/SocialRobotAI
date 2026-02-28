from deepface import DeepFace
import cv2

interval = 5
batch_size = 8
# List[pd.DataFrame] = DeepFace.find(img_path = "img1.jpg", db_path = "C:/my_db")


class DetectFace:
    def __init__(self):
        # self.frame = frame
        pass

    print("Pri deepface smo prisli not")
    trackers = []

    def faceDetection(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend="opencv")

            multiple_bboxes = []
            if faces is not None:
                # for f in faces:
                #     fa = f["facial_area"]
                #     print(f"Found facial area: {fa}")
                #     x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                #     # return x, y, w, h
                for detected_face in faces:
                    # dostopaj do dobljenih koordinat iz detektiranga obraza izmed vseh detektiranih
                    fa = detected_face["facial_area"]

                    print("Detected face keys:", detected_face.keys())

                    face_cropped = detected_face["face"]

                    emo_result = DeepFace.analyze(
                        face_cropped, actions=["emotion"], enforce_detection=False
                    )

                    print("Detected emotions:", emo_result.get("emotion"))
                    print("Dominant emotion:", emo_result.get("dominant_emotion"))
                    # detected_emotion = detected_face["dominant_emotion"]
                    # print(f"detected emotion: {detected_emotion}")

                    multiple_bboxes.append((fa["x"], fa["y"], fa["w"], fa["h"]))
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
