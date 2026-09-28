import os
from deepface_detection_mp import process_handler
import multiprocessing as mproc
import queue
import time
import joblib
import pandas as pd
# import torch
# import torchvision
# import threading
# import queue

class ProcessFrames:
    def __init__(self, video_path, req_q, res_q, stop_event, detection_interval=10, frameId=0):
        self.video_path = video_path
        self.detection_interval = detection_interval  # na vsake N frameov še enkrat poišče nove obraze, da se lahko trackeri posodobijo, če se osebe premikajo ali pridejo nove v frame
        self.last_detected_boxes = ([])  # buffer za ground truth bboxe, če dobim conf 0 gre čez celoten frame ?
        self.trackers = ([])  # tracker buffer, shrani vse trackerje, ki jih imamo trenutno aktivne (za vsako zaznano osebo)
        self.person_id = 0  # za vsako novo zaznano osebo, da ji damo ID, da lahko trackamo isto osebo čez več frameov

        self.frameId = frameId

        # tole baje ni ok za moj init za mp workerja? kam nej jih dam??
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(
        #     min_detection_confidence=0.5, min_tracking_confidence=0.4
        # )  # kakšen naj bo optimalen confidence tu?

        self.request_queue = req_q
        self.result_queue = res_q
        self.stop_event = stop_event #a tega rabim tu al ne?

        self.detection_process = None
        self._request_in_flight = False
        self.face_confidence = None

        self.bbox_x = None
        self.bbox_y = None
        self.bbox_h = None
        self.bbox_w = None


    
    def initialize_tracker(self, frame, bbox):
        tracker = cv2.legacy.TrackerCSRT_create() #al cv2.legacy.TrackerMOSSE_create()
        bbox = tuple(map(int, bbox))  # bbox v (x, y, w h) formatu
        ok = tracker.init(frame, bbox)
        # print("Tracker init:", ok)
        # print("BBox:", bbox)
        return tracker

    def collect_mprocess_detection_result(self, currentframe, frameId):
        #print("Collection deepface multiprocess results...................")
        try:
            frame_id, detected_bboxes = self.result_queue.get_nowait() #!!!TODO a dam nowait pa a nej tu dostopam do result queue preko self al ga pošljem prek funkcij?
            #print("🛠️Probam collectat iz mprocessa pri frameid iz MPJA: ", frame_id)
        except queue.Empty:
            #TODO tu popravi k vmes skos to izpisuje da je queue empty?
            #print("Queue je empty pri collectionu......")
            return
        # self._request_in_flight = False #TODO ta flag nazaj implementiraj pravilno
        #print(f"Iz mprocess smo dobili parametre: {detected_bboxes}", "kjer so pri main pipeline frameu: ", frameId)

        if detected_bboxes:
            x, y, w, h = detected_bboxes[0]["bbox"]
            conf_score = detected_bboxes[0]["conf"]

            #In za globalen score:
            self.face_confidence = detected_bboxes[0]["conf"]
            self.bbox_x = x
            self.bbox_y = y
            self.bbox_h = h
            self.bbox_w = w

            detected_emotion = detected_bboxes[0]["emo"]

            new_tracker = self.initialize_tracker(currentframe, (x, y, w, h))
            
            detection_conf_score = "EMO: " + str(detected_emotion)

            cv2.rectangle(
                currentframe, (x, y), (x + w, y + h), (255, 255, 128), 1
            )
            cv2.putText(
                        currentframe,
                        detection_conf_score,
                        (x + w, (y-20) + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 128),
                        1
                    )

            self.trackers = [new_tracker]
            # self.last_label = detected_bboxes[0]["emo"] or "not detected"
            # self.last_conf = detected_bboxes[0]["conf"]
        else:
            print("No boundingbox given from multiprocess of deepface")

    # poslji frame v deepface multiprocess
    def request_mprocess_detection(self, currentframe, frameId):
        # currentN = (frameId % self.detection_interval == 0) or len(self.trackers) == 0
        if frameId % self.detection_interval != 0: #TODO and not self.request_in_flight?
            # if not currentN or self._request_in_flight:
            #print("V requestu se returna, ni N-ti frame: ", frameId)
            return
        try:
            #print("🛠️ puttam nowait v request queue za frame: ", frameId)
            self.request_queue.put_nowait((frameId, currentframe.copy())) #TODO put z nowait al brez pa a timeout al ne?
            # self._request_in_flight = False #TODO to gre nazaj na false če smo izpolnili request?
        except queue.Full:
            print("Worker je ševedno busy")
            pass  # worker still busy on a previous frame - skip this request

    def process_frames(self):
        import cv2
        import mediapipe as mp
        import numpy as np
        # tole dej v init
        # self.df = DetectFace()
        cap = cv2.VideoCapture(self.video_path)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.4
        )  # kakšen naj bo optimalen confidence tu?
        
        frameId = 0
        last_timestamp = 0
        while cap.isOpened():
            ret, currentframe = cap.read()
            if not ret:
                print("Napaka pri zajemanju videa ali EOF videa")
                break
            
            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) #tud nedela
            # fps = cap.get(cv2.CAP_PROP_FPS) #nedela
            # print(f"TIME JE: {time_s:.2f} s")
            # print(f"ŠT FRAMEOV JE: {frame_count} frames")
            # print(f"fps JE: {fps} frames")
            
            # TODO a morem to kam drgam postavit?
            self.collect_mprocess_detection_result(currentframe, frameId)
            self.request_mprocess_detection(currentframe, frameId)

            # mediapipe variables for pose landmarks
            rgb_frame = cv2.cvtColor(currentframe, cv2.COLOR_BGR2RGB)
            
            #pose mesh
            results_pose = self.pose.process(rgb_frame)
            pose_landmarks = (
                results_pose.pose_landmarks.landmark
                if results_pose.pose_landmarks
                else None
            )

            pitch, yaw, roll = 0, 0, 0
            pose_detected_this_frame = False
            h_img, w_img, _ = currentframe.shape

            if pose_landmarks:
                pose_detected_this_frame = True
                nose = pose_landmarks[0] #nose
                #tele vse so obrnjene glede na pogleda iz kamere, v nasprotnem primeru obrni:
                left_eye = pose_landmarks[5] #left eye
                right_eye = pose_landmarks[2] #right eye
                #print("LEFT EYE COORDS: ", left_eye.x, left_eye.y, "RIGHT EYE COORDS: ", right_eye.x, right_eye.y)
                
                left_ear = pose_landmarks[8] #left ear
                right_ear = pose_landmarks[7] #right ear
                # print("LEFT EAR COORDS: ", left_ear.x, left_ear.y, "RIGHT EAR COORDS: ", right_ear.x, right_ear.y)
                
                left_mouth = pose_landmarks[10] #left mouth
                right_mouth = pose_landmarks[9] #right mouth
                #mouth_center = (left_mouth + right_mouth) / 2.0          

                #print("LEFT MOUTH COORDS: ", left_mouth.x, left_mouth.y, "RIGHT MOUTH COORDS: ", right_mouth.x, right_mouth.y)
                ####vsi te zgornji coordinates so normalizirani od mediapipa med 0 pa 1, t

                # pts = np.array(
                #         [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                #         dtype=np.float32,
                #     )

                # def p(idx):
                #         return pts[idx]

                # --- face geometry ---
                # face_center = pts[[0, 1, 33, 61, 291, 199]].mean(axis=0)
                # nose = p(1)
                # chin = p(152)
                # left_eye_corner = p(33)
                # right_eye_corner = p(263)
                # left_ear = p(234)
                # right_ear = p(454)

                #######======= poskus convertat image coords iz pose landmarks za solvepnp da dobim euler angles
                # h_img, w_img, _ = currentframe.shape ta je brez uses
                # image_points = np.array([
                #     [nose.x * w_img, nose.y * h_img],
                #     [left_eye.x * w_img, left_eye.y * h_img],
                #     [right_eye.x * w_img, right_eye.y * h_img],
                #     [left_mouth.x * w_img, left_mouth.y * h_img],
                #     [right_mouth.x * w_img, right_mouth.y * h_img],
                # ], dtype=np.float64)
                
                image_points = np.array([
                    [nose.x * w_img,       nose.y * h_img],        # 0 nose
                    [left_eye.x * w_img,   left_eye.y * h_img],    # 2 left eye
                    [right_eye.x * w_img,  right_eye.y * h_img],   # 5 right eye
                    [left_ear.x * w_img,   left_ear.y * h_img],    # 7 left ear
                    [right_ear.x * w_img,  right_ear.y * h_img],   # 8 right ear
                    [left_mouth.x * w_img, left_mouth.y * h_img],  # 9 left mouth
                    [right_mouth.x * w_img,right_mouth.y * h_img], # 10 right mouth
                ], dtype=np.float64)

                model_points = np.array([
                    [0.0,   0.0,   0.0],     # nose
                    [-3.0,  2.0,  -2.0],     # left eye
                    [3.0,   2.0,  -2.0],     # right eye
                    [-5.0,  1.0,   0.0],     # left ear
                    [5.0,   1.0,   0.0],     # right ear
                    [-2.5, -3.0,  -1.0],     # left mouth
                    [2.5,  -3.0,  -1.0],     # right mouth
                ], dtype=np.float64)

                #matrika kamere 
                focal_length = w_img
                camera_matrix = np.array([
                    [focal_length, 0, w_img / 2],
                    [0, focal_length, h_img / 2],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4, 1), dtype=np.float64)
                
                #zazaeni solvepnp ki vrne rodriguez
                success, rvec, tvec = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                #convert rodriguez to rotation matrix
                #rotation_matrix, _ = cv2.Rodrigues(rvec) #matrika orientacij ki jo je ocenu pnp
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                    pitch = np.radians(euler_angles[0])
                    yaw   = np.radians(euler_angles[1])
                    roll  = np.radians(euler_angles[2])

                #######============================
                #roll na star nacin
                #roll = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
                # left_eye = np.array([left_eye.x, left_eye.y])
                # right_eye = np.array([right_eye.x, right_eye.y])
                # roll = np.arctan2(
                #     right_eye[1] - left_eye[1],
                #     right_eye[0] - left_eye[0]
                # )

                #proposed glede na moje vrednosti kernimam facemesha
                # roll = np.arctan2(
                #     right_eye_y - left_eye_y,
                #     right_eye_x - left_eye_x
                # )
                # eye_mid_x = (left_eye.x + right_eye_x) / 2
                # eye_mid_y = (left_eye_y + right_eye_y) / 2
                
                #pitch yaw roll, ty random indian
                # nose_tip = landmarks[1]
                # chin = landmarks[152]
                # left_eye = landmarks[33]
                # right_eye = landmarks[263]
                # roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                # pitch = np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2])
                # yaw = np.arctan2(nose_tip[0] - chin[0], nose_tip[2] - chin[2])
            
            if self.trackers:
                #print("Not N-th frame, skipping detection...")
                ok, bbox = self.trackers[0].update(currentframe)
                #print(f"Tracker update result: {ok}, bbox: {bbox}")
                if ok:
                    x, y, w, h = map(int, bbox)
                    #izlusci koordinate obraza in mediapipe koordinate za featurje za attentiveness SVM model
                    face_x = x
                    face_y = y
                    face_w = w
                    face_h = h

                    left_eye_px  = np.array([left_eye.x  * w_img, left_eye.y  * h_img])
                    right_eye_px = np.array([right_eye.x * w_img, right_eye.y * h_img])
                    nose_px      = np.array([nose.x * w_img, nose.y * h_img])
                    mouth_px     = np.array([(left_mouth.x + right_mouth.x) / 2 * w_img,
                                            (left_mouth.y + right_mouth.y) / 2 * h_img])

                    eye_mid = (left_eye_px + right_eye_px) / 2
                    iod = np.linalg.norm(right_eye_px - left_eye_px)

                    if pose_detected_this_frame and iod > 1e-6:
                        left_eye_x_norm, left_eye_y_norm   = (left_eye_px  - eye_mid) / iod
                        right_eye_x_norm, right_eye_y_norm = (right_eye_px - eye_mid) / iod
                        nose_tip_x_norm, nose_tip_y_norm   = (nose_px       - eye_mid) / iod
                        mouth_x_norm, mouth_y_norm         = (mouth_px      - eye_mid) / iod

                        svm_features = {
                            "left_eye_x": left_eye_x_norm, 
                            "left_eye_y": left_eye_y_norm,
                            "right_eye_x": right_eye_x_norm, 
                            "right_eye_y": right_eye_y_norm,
                            "nose_tip_x": nose_tip_x_norm, 
                            "nose_tip_y": nose_tip_y_norm,
                            "mouth_x": mouth_x_norm, 
                            "mouth_y": mouth_y_norm,
                            # "head_pitch": pitch, 
                            # "head_yaw": yaw, 
                            # "head_roll": roll
                        }
                        svm_ft_pd = pd.DataFrame([svm_features])
                        print(svm_features)
                        #print(svm_ft_pd.describe())
                    else:
                        continue

                    X = pd.DataFrame([svm_features])

                    # print("coords: ", X.T)
                    # print("more coords: ", {
                    #     "face": [x, y, w, h],
                    #     "left_eye": [left_eye_x, left_eye_y],
                    #     "right_eye": [right_eye_x, right_eye_y],
                    #     "nose": [nose_tip_x, nose_tip_y],
                    #     "mouth": [mouth_x, mouth_y],
                    #     "pitch": pitch,
                    #     "yaw": yaw,
                    #     "roll": roll,
                    # })

                    attention_text = "Not classified"
                    
                    prediction_label = model.predict(X)[0]
                    probs_conf = model.predict_proba(X)[0]
                    
                    prediction_confidence = model.predict_proba(X).max()

                    print(
                        f"class 0 = {probs_conf[0]:.6f}, "
                        f"class 1 = {probs_conf[1]:.6f}"
                    )
                    
                    if prediction_confidence is not None:
                        if prediction_label == 0:
                            attention_text = "Attentive:{:.6f}".format(prediction_confidence)
                        else:
                            attention_text = "Inattentive:{:.6f}".format(prediction_confidence)

                    cv2.rectangle(
                        currentframe, (x, y), (x + w, y + h), (255, 255, 0), 2
                    )
                    cv2.putText(
                        currentframe,
                        attention_text,
                        (x + w, (y + 10) + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )
                else:
                    print("Tracker lost person")
            else:
                print(f"[frame {frameId}] pose FAILED - reusing stale landmarks")       

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

            # print(f"New detected boxes: {new_detected_boxes}")

            self.mp_drawing.draw_landmarks(
                currentframe, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            # ustvari row za dataframe časovnih vrst
            # rows.append(row)
            cv2.imshow("Video", currentframe)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows() # a to rabim tu al v finally, pa dam tu samo break?

            frameId += 1
            last_timestamp = time_s
        # finally: #a rabm to tuki??? ne če je v mainu?? TODO
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     if self.detection_process is not None:
        #         self.mp_stop.set()
        #         # self.detection_process.join(timeout=2)
        #         self.detection_process.join(timeout=3)
        #         if self.detection_process.is_alive():
        #             self.detection_process.terminate()
        # df_timeseries = pd.DataFrame(rows)
        # return df_timeseries
        return frameId, last_timestamp


if __name__ == "__main__":
    import cv2
    mproc.set_start_method("spawn", force=True)
    request_queue = mproc.Queue(maxsize=1)
    result_queue = mproc.Queue(maxsize=1)
    
    ready_event = mproc.Event()
    stop_event = mproc.Event()
    
    worker = mproc.Process(
        target=process_handler,
        args=(request_queue, result_queue, stop_event, ready_event),
        daemon=True
    )
    video_files = []
    for root, dirs, files in os.walk("Data"):
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and file.lower().endswith(
                ".webm"
            ):
                print("Found videofile: ", file)
                video_files.append(os.path.join(root, file))
    
    worker.start()
    ready_event.wait()
    csv_id = 0
   
    model = joblib.load("attention_svm.joblib")  #model za klasifikacijo

    try:
        for video in video_files:
            # cap = cv2.VideoCapture(video)
            # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # trajanje_posnetka = total_frames / fps
            # cap.release()

            start_time = time.perf_counter()
            pf = ProcessFrames(video, request_queue, result_queue, stop_event)
            processed_frames, trajanje = pf.process_frames()
            
            end_time = time.perf_counter()

            trajanje_videa = processed_frames / trajanje
            
            total_time = end_time - start_time
            zaostanek = total_time - trajanje_videa

            print(f"za Video: {video}")
            print(f"Trajanje: {trajanje} sekund")
            print(f"Število prebranih okvirjev: {processed_frames} okvirjev")
            print(f"Trajanje videa: {trajanje_videa} sekund")
            # print(f"Število okvirjev: {total_frames} okvirjev")
            # print(f"FPS videa: {fps} s")
            print(f"Total processing time: {total_time:.2f} s")
            print(f"Processing delay: {zaostanek:.2f} s")
            # df.to_csv(f"timeseries_{csv_id}.csv", index=False)
            # print(f"Saved {len(df)} frames to timeseries.csv from video {video}")
            csv_id += 1
    finally:
        stop_event.set()
        worker.join() #al brez timeouta,al kok nej dam?
        if worker.is_alive():
            worker.terminate()
            worker.join() #a ta join je tu potreben al mam sam tega pred ifom?

    # vivit = False  # flag for vivit execution
    # # emotion_model="mobilenetv3
    # if not vivit:
    #     ############ PROCESS FRAMES FOR EACH VIDEO ############
    #     csvid = 0
    #     for video in video_files:
    #         pf = ProcessFrames(video)
    #         df = pf.process_frames()
    #         df.to_csv(f"timeseries_{csvid}.csv", index=False)
    #         print(f"Saved {len(df)} frames to timeseries.csv from video {video}")
    #         csvid += 1
    # else:
    #     from vivit_transformer import (
    #         train_and_evaluate,
    #         model,
    #         train_loader,
    #         val_loader,
    #     )
    #     train_and_evaluate(model, train_loader, val_loader, num_epochs=10)
