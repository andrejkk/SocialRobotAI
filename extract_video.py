import os
import re
from pathlib import Path
import cv2

import mediapipe as mp
from deepface_detection import DetectFace

"""
STREAMING PIPELINE
"""


class ProcessFrames:
    def __init__(self, video_path):
        self.video_path = video_path

    def process_frames(self):
        df = DetectFace()
        cap = cv2.VideoCapture(self.video_path)
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        trackers = []
        frameId = 0
        while cap.isOpened():
            print(f"Processing frame {frameId}")
            ret, currentframe = cap.read()
            if not ret:
                print("Napaka pri zajemanju videa")
                break

            rgb_pose = cv2.cvtColor(currentframe, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_pose)

            detected_boxes = df.faceDetection(currentframe)
            if detected_boxes is None:
                # ignoriraj prazne rezulte, pojdi na naslednji okvir
                continue

            for x, y, w, h in detected_boxes:
                cv2.rectangle(currentframe, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
            cv2.imshow("Output", currentframe)
            # cv2.imshow("Video", currentframe)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                # out.release()
                cv2.destroyAllWindows()

            frameId += 1
            # if frameId >= 100:
            #     break
            # else:
            #     continue


if __name__ == "__main__":
    patternpath = re.compile(r"Data\/66\d{3}\/\d{4}-\d{2}-\d{2}\/S\d+\/[^.]+\.webm$")
    # video_dir = Path("Data")
    # test = "Data/66001/2023-11-01/S1/f565c08a-5fb8-41c5-9da7-e2a5dc1a6af8.webm"
    # print(patternpath)
    # print(bool(patternpath.match(test)))

    # videos = [
    #     f for f in video_dir.iterdir() if f.is_dir() and patternpath.match(f.name)
    # ]
    videos = []
    root = Path("Data")
    match_posix = [p for p in root.glob("**/*.webm") if patternpath.match(str(p))]
    video_string = [str(p) for p in match_posix]

    # cap = cv2.VideoCapture(video_string[0])

    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for video in video_string:
        print(video)
    pf = ProcessFrames(video_string[1])
    pf.process_frames()

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
