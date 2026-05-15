import os
import cv2
import pandas as pd
import mediapipe as mp
from deepface_detection import DetectFace
import torch
import torchvision

# import re
# from pathlib import Path

"""
STREAMING PIPELINE
TODO :
1. emotional annotation iz deepface.analyze [ x ]
2. daj pose estimation v posebaj file da se klice kot razred, razreši ga na windowsu [ ]
3. implementiraj trackerje preko bounding boxov, pridobi IoU za updatanje in deletanje trackerjev [ ]
4. najti javno dostopne video posnetke z označenimi dogodki, da se lahko testira na realnih podatkih [ ] 
(poglej WESAD, AffectiveROAD, RECOLA, SEMAINE, DEAP, AMIGOS, HCI Tagging Database)

Na kratko:
- potrebujemo ekstrahirane časovne vrste, da gredo v obdelavo. WESAD in podobnih baz ne potrebujeva
- predlagam, da poiščete po javno dostopih podatkovnih bazah označenih videov, za katere so dogodki znani in bo to eden od podatkovnih množic v mag. 
Gre za video posnetke, na katerih so osebe v interakciji z nekim sistemom in so označeni dogodki.

5. data loading pipeline za transformer
6. queue za threading frameov
7. 
"""


class ProcessFrames:
    def __init__(self, video_path, detection_interval=5):
        self.video_path = video_path
        self.detection_interval = detection_interval
        self.trackers = []  # tracker buffer
        self.last_detected_boxes = []

    def get_iou(ground_truth, pred):
        gt_box_tensor = torch.tensor([ground_truth], dtype=torch.float32)
        pred_box_tensor = torch.tensor([pred], dtype=torch.float32)
        iou = torchvision.ops.box_iou(gt_box_tensor, pred_box_tensor)
        return iou.item()

    def update_trackers(self, trackers, frame, last_detected_boxes):
        for i, tracker in enumerate(trackers):
            # try:
            #     success, roi = tracker.update(frame)
            # except Exception:
            #     success = False
            #     roi = None

            ok, bbox = tracker.update(frame)  # bbox = (x, y, w, h) (floats)
            if ok:
                x, y, w, h = map(int, bbox)
                # crop = frame[y : y + h, x : x + w]
                # if success and roi is not None:
                # x, y, w, h = last_detected_boxes["bbox"]
                # x, y, w, h = map(int, roi) kako to uporabim?? TODO
                cropped_face = frame[y : y + h, x : x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # emotion = last_detected_boxes[i][4] if (i < len(last_detected_boxes) and last_detected_boxes[i]) else None

                label = None
                if i < len(last_detected_boxes):  # and last_detected_boxes[0]
                    label = last_detected_boxes["emo"]
                if label:
                    cv2.putText(
                        frame,
                        label,
                        (x + w, (y + 10) + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            else:
                print(f"Tracker za osebo {i} ni bil zaznan.")

    def process_frames(self):
        # CONFIDENCE_THRESHOLD = 0.8 #za yolo uporabi
        ### Init DeepFace df detection with CV2, MediaPipe pose
        df = DetectFace()
        cap = cv2.VideoCapture(self.video_path)

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # trackers = []
        # last_detected_boxes = []

        ### Init frame counter and cv2 tracker for tracking detected faces across frames
        frameId = 0
        rows = []

        while cap.isOpened():
            ret, currentframe = cap.read()
            new_detected_boxes = None
            if not ret:
                print("Napaka pri zajemanju videa")
                break

            # mediapipe variables for pose landmarks
            rgb_pose = cv2.cvtColor(currentframe, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_pose)
            pose_landmarks = (
                results_pose.pose_landmarks.landmark
                if results_pose.pose_landmarks
                else None
            )

            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Process every 5 frames or if we have no trackers
            if frameId % self.detection_interval == 0 or not self.trackers:
                detected_bboxes = df.faceDetection(currentframe)
                # ubistvu 1 box s koordinatamo x y w h?
                print(f"Dobili smo dict bboxov: {detected_bboxes}")

                # če smo dobili nove bboxe iz detekcije,
                # jih uporabimo in shranimo za naslednje frame-e, sicer uporabimo prejšnje bboxe (če obstajajo)
                ### CREATE CV2 TRACKER FROM NEW BBOX
                if detected_bboxes is not None:
                    t = cv2.TrackerKCF_create()
                    # t = cv2.TrackerCSRT_create()
                    print("Koordinate za prvi tracker so: ", detected_bboxes[0]["bbox"])

                    t.init(currentframe, detected_bboxes[0]["bbox"])
                    self.trackers.append(t)

                    x, y, w, h = detected_bboxes[0]["bbox"]
                    print(
                        f"Detected face at ({x}, {y}, {w}, {h}) with emotion: {detected_bboxes[0]['emo']}"
                    )

                    new_detected_boxes = detected_bboxes[0]
                    self.last_detected_boxes = new_detected_boxes
                ### UPDATE TRACKERS FROM PREVIOUS BBOXES
                else:
                    self.last_detected_boxes = new_detected_boxes
                    self.update_trackers(
                        self.trackers, currentframe, new_detected_boxes
                    )

                    print("Detection failed this frame; using last detected boxes.")

            # ### Run detector only every N frames (or if we have no trackers)
            # detected_boxes = None
            # if frameId % self.detection_interval == 0 or not self.trackers:
            #     det_res = df.faceDetection(currentframe)
            #     if det_res is not None:
            #         detected_boxes, trackers = det_res
            #         # keep last detected boxes for labeling while tracking
            #         self.last_detected_boxes = detected_boxes if detected_boxes else []
            #         self.trackers = trackers if trackers else []
            #     else:
            #         # detection failed this frame; keep previous trackers/boxes
            #         detected_boxes = None

            # # prefer emotion/coords from current detection; fall back to last detected boxes
            # if detected_boxes:
            #     x, y, w, h, emotion = detected_boxes[0]
            # elif self.last_detected_boxes:
            #     x, y, w, h, emotion = self.last_detected_boxes[0]
            # else:
            #     x = y = w = h = emotion = None

            # --- ZGRADI ROW ---
            row = {
                "time_s": time_s,
                "emotion": "not given",
                "face_x": 0,
                "face_y": 0,
                "face_w": 0,
                "face_h": 0,
            }

            if pose_landmarks:
                h_img, w_img, _ = currentframe.shape
                for li, lm in enumerate(pose_landmarks):
                    # norm x y to image size
                    row[f"pose_{li}_x"] = int(lm.x * w_img)
                    row[f"pose_{li}_y"] = int(lm.y * h_img)
                    row[f"pose_{li}_z"] = lm.z

            # --- ADD POSE (example: only nose landmark) ---
            # if pose_landmarks:
            #     nose = pose_landmarks[0]  # landmark 0 = nose
            #     h_img, w_img, _ = currentframe.shape

            #     row["nose_x"] = int(nose.x * w_img)
            #     row["nose_y"] = int(nose.y * h_img)
            # else:
            #     row["nose_x"] = None
            #     row["nose_y"] = None

            # if pose_landmarks:
            #     for li, lm in enumerate(pose_landmarks):
            #         px = int(lm.x * CV2.CAP_PROP_FRAME_WIDTH)
            #         py = int(lm.y * CV2.CAP_PROP_FRAME_HEIGHT)
            #         pz = lm.z  # depth is relative; keep as-is
            #         row[f"pose_{li}_x"] = px
            #         row[f"pose_{li}_y"] = py
            #         row[f"pose_{li}_z"] = pz
            # else:
            #     # if missing, add None placeholders (optional)
            #     # for li in range(33): row[f"pose_{li}_x"] = row[f"pose_{li}_y"] = row[f"pose_{li}_z"] = None
            #     pass

            print(f"New detected boxes: {new_detected_boxes}")
            print(f"Trackers: {self.trackers}")

            # Update trackers on non-detection frames and draw boxes/labels
            self.update_trackers(self.trackers, currentframe, self.last_detected_boxes)
            ###update trackers here

            mp_drawing.draw_landmarks(
                currentframe, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            # ustvari row za dataframe časovnih vrst
            # rows.append(row)
            cv2.imshow("Video", currentframe)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()

            frameId += 1
        # df_timeseries = pd.DataFrame(rows)
        # return df_timeseries
        return []


if __name__ == "__main__":
    # test = "Data/66001/2023-11-01/S1/f565c08a-5fb8-41c5-9da7-e2a5dc1a6af8.webm"
    ############ FIND VIDEO FILES ############
    # 1. poiščemo vse video posnetke v Data mapi (lahko tudi globoko po submapah) - recimo samo .webm
    # 2. kličemo ProcessFrames, za vsake N frameov kličemo detektor, ki vrne emotion in koordinate obraza
    #
    #

    video_files = []
    for root, dirs, files in os.walk("Data"):
        print("Searching for files in: ", root)
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and file.lower().endswith(
                ".webm"
            ):
                # print("Found videofile: ", file)
                video_files.append(os.path.join(root, file))

    ############ PROCESS FRAMES FOR EACH VIDEO ############
    csvid = 0
    for video in video_files:
        pf = ProcessFrames(video)
        df = pf.process_frames()
        df.to_csv(f"timeseries_{csvid}.csv", index=False)
        print(f"Saved {len(df)} frames to timeseries.csv from video {video}")
        csvid += 1

    # pf = ProcessFrames(video_files[0])
    # pf.process_frames()

    # while cap.isOpened():

    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     # frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    #     # Display the resulting frame
    #     cv2.imshow("Frame", frame)

    # for path in root.glob("*.webm"):
    #     if patternpath.match(str(path)):
    #         videos.append(path)
    # print(videos)

    # output_dir = "frames/"

    # extractor = VideoExtractor(video_string[0], output_dir)
    # extractor.extract_frames()

    ##### extract frames #####
    # cap = cv2.VideoCapture(video_string[0])
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # out_video_path = "output_video.mp4"

    # frameId = 0
    # while frameId < 100:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # cv2.imwrite(f"frames/frame_{frameId:04d}.jpg", frame)

    #     frameId += 1
