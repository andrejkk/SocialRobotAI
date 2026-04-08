import os
import re
from pathlib import Path
import cv2
import pandas as pd

import mediapipe as mp
from deepface_detection import DetectFace

"""
STREAMING PIPELINE

TODO ***do 13. marca***:
1. emotional annotation iz deepface.analyze [ x ]
2. daj pose estimation v posebaj file da se klice kot razred, razreši ga na windowsu [ ]
3. implementiraj trackerje preko bounding boxov, threadaj jih? [ ]
4. najti javno dostopne video posnetke z označenimi dogodki, da se lahko testira na realnih podatkih [ ] 
(poglej WESAD, AffectiveROAD, RECOLA, SEMAINE, DEAP, AMIGOS, HCI Tagging Database)

--3. marec--

Na kratko:
- sedaj potrebujemo ekstrahirane časovne vrste, da gredo v obdelavo. WESAD in podobnih baz ne potrebujeva
- predlagam, da poiščete po javno dostopih podatkovnih bazah označenih videov, za katere so dogodki znani in bo to eden od podatkovnih množic v mag. 
Gre za video posnetke, na katerih so osebe v interakciji z nekim isstemom in so označeni dogodki.
- struktura dokumenta: za to potrebujeva sestanek, da zastaviva poglavja in okvir vsebine - na daljavo z deljejnem zaslona
-----------

"""


class ProcessFrames:
    def __init__(self, video_path):
        self.video_path = video_path

    def process_frames(self):
        ### Init DeepFace and cv2 capture
        df = DetectFace()
        cap = cv2.VideoCapture(self.video_path)

        ### Init MediaPipe Pose
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        trackers = []

        ### Init frame counter and cv2 tracker for tracking detected faces across frames
        tracker = cv2.TrackerCSRT_create()
        frameId = 0
        rows = []
        # frames = self.video_path.get(cv2.CAP_PROP_FRAME_COUNT)

        while cap.isOpened():

            # print(f"Processing frame {frameId} out of frames = {frames}")
            # print(f"Processing frame {frameId}")
            ret, currentframe = cap.read()
            if not ret:
                print("Napaka pri zajemanju videa")
                break

            rgb_pose = cv2.cvtColor(currentframe, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_pose)

            # landmarks pose
            pose_landmarks = (
                results_pose.pose_landmarks.landmark
                if results_pose.pose_landmarks
                else None
            )

            ###TODO: TypeError: cannot unpack non-iterable NoneType object za oba? mogoče zarad kakšnega none od trackerja ali bboxa?
            detected_boxes, trackers = df.faceDetection(currentframe)
            if detected_boxes is None or trackers is None:
                # ignoriraj prazne rezulte, pojdi na naslednji okvir
                continue

            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # rows.append(
            #     {
            #         "time_s": time_s,
            #         "emotion": detected_boxes[0][4] if detected_boxes else None,
            #         "face_x": detected_boxes[0][0] if detected_boxes else None,
            #         "face_y": detected_boxes[0][1] if detected_boxes else None,
            #         "face_w": detected_boxes[0][2] if detected_boxes else None,
            #         "face_h": detected_boxes[0][3] if detected_boxes else None,
            #     }
            # )
            if detected_boxes:
                x, y, w, h, emotion = detected_boxes[0]
            else:
                x = y = w = h = emotion = None

            # --- BUILD ROW ---
            row = {
                "time_s": time_s,
                "emotion": emotion,
                "face_x": x,
                "face_y": y,
                "face_w": w,
                "face_h": h,
            }

            # Populate pose landmarks into the row. Use actual frame dimensions
            # and avoid the `for-else` pattern which was overwriting values.
            if pose_landmarks:
                h_img, w_img, _ = currentframe.shape
                for li, lm in enumerate(pose_landmarks):
                    # convert normalized x,y to pixels using image size
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

            print(f"Detected boxes: {detected_boxes}")
            print(f"Trackers: {trackers}")

            for i, tracker in enumerate(trackers):
                success, roi = tracker.update(currentframe)
                if success:
                    x, y, w, h = map(int, roi)
                    cv2.rectangle(currentframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        currentframe,
                        detected_boxes[i][4],
                        (x + w, (y + 10) + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                else:
                    print(f"Tracker za osebo {i} ni bil zaznan.")

            # for x, y, w, h, emotion in detected_boxes:
            #     init_tracker = tracker.init(currentframe, (x + w, y + h))

            #     cv2.rectangle(currentframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     cv2.putText(
            #         currentframe,
            #         emotion,
            #         (x + w, (y + 10) + h),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 255, 0),
            #         2,
            #     )

            # pokaži pridobljeno cropped sliko
            # cropped = currentframe[y : y + h, x : x + w]
            # cv2.imshow("Cropped Face", cropped)

            ## init a tracker on retrieved bbox, true if succesful on the bb for target person
            # cv.Tracker.init(	image, boundingBox	)

            # updateat tracker da najde nov most likely bb od targeta
            # cv.Tracker.update(	image	)

            # tracker = cv2.TrackerCSRT_create()
            # else:
            #     for tracker in trackers:
            #         success, bbox = tracker.update(currentframe)
            #         if success:
            #             x, y, w, h = map(int, bbox)
            #             cv2.rectangle(currentframe, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.imwrite(f"frames/frame_{frameId:04d}.jpg", frame)
            mp_drawing.draw_landmarks(
                currentframe, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            # ustvari row!
            rows.append(row)
            cv2.imshow("Video", currentframe)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()

            frameId += 1
        df_timeseries = pd.DataFrame(rows)
        return df_timeseries


if __name__ == "__main__":
    # test = "Data/66001/2023-11-01/S1/f565c08a-5fb8-41c5-9da7-e2a5dc1a6af8.webm"
    ############ FIND VIDEO FILES ############
    video_files = []
    for root, dirs, files in os.walk("Data"):
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and file.lower().endswith(
                ".webm"
            ):
                print(file)
                video_files.append(os.path.join(root, file))

    ### deprecated regex pattern, ne dela na Windowsu ker glob vrne POSIXPath, ki ima naprej poševnice
    # patternpath = re.compile(r"Data\/66\d{3}\/\d{4}-\d{2}-\d{2}\/S\d+\/[^.]+\.webm$")
    # video_dir = Path("Data")

    # # print(patternpath)
    # # print(bool(patternpath.match(test)))

    # # videos = [
    # #     f for f in video_dir.iterdir() if f.is_dir() and patternpath.match(f.name)
    # # ]
    # # videos = []
    # root = Path("Data/")
    # match_posix = [p for p in root.glob("**/*.webm") if patternpath.match(str(p))]
    # video_string = [str(p) for p in match_posix]
    # print('match_posix =', match_posix)
    # cap = cv2.VideoCapture(video_string[0])

    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # for video in video_string:
    #    print(video)
    ###############################################################
    ### walk through all found videos and process their frames
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
