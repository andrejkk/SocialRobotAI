import os
from deepface_detection import process_handler
import multiprocessing as mproc
import queue
# import torch
# import torchvision

# import threading
# import queue
"""
STREAMING PIPELINE
TODO :
1. emotional annotation iz deepface.analyze [ x ] ALI FER?
2. daj pose estimation v posebaj file da se klice kot razred, razreši ga na windowsu [ ]
3. implementiraj trackerje preko bounding boxov, pridobi IoU za updatanje in deletanje trackerjev [ ]
4. najti javno dostopne video posnetke z označenimi dogodki, da se lahko testira na realnih podatkih [ ] 
(poglej WESAD, AffectiveROAD, RECOLA, SEMAINE, DEAP, AMIGOS, HCI Tagging Database)

Na kratko:
- potrebujemo ekstrahirane časovne vrste, da gredo v obdelavo. WESAD in podobnih baz ne potrebujeva
- predlagam, da poiščete po javno dostopih podatkovnih bazah označenih videov, za katere so dogodki znani in bo to eden od podatkovnih množic v mag. 
Gre za video posnetke, na katerih so osebe v interakciji z nekim sistemom in so označeni dogodki.

5. data loading pipeline za transformer
6. queue IN za mutluthreading frameov - deadlock UPOŠTEVAJ NUJNO!! [ X ]
7. izberi tapravi tracker! [ X ] CSRT dela
8. fixaj 0 confidence bounding box za detector backend ki ni opencv [ X ] (mogoče upscaleat za druge )
"""
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

    def initialize_tracker(self, frame, bbox):
        tracker = cv2.legacy.TrackerCSRT_create() #al rab cv2.legacy.TrackerMOSSE_create() ??
        bbox = tuple(map(int, bbox))  # Ensure bbox is in (x, y, w, h) format
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
            new_tracker = self.initialize_tracker(currentframe, (x, y, w, h))
            
            detection_conf_score = "RE-INITED BBOX WITH" + str(conf_score)

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

            self.trackers = [new_tracker]  # same reset-to-one behavior as before
            # self.last_label = detected_bboxes[0]["emo"] or "not detected"
            # self.last_conf = detected_bboxes[0]["conf"]
        else:
            print("No boundingbox given from multiprocess of deepface")

    # send frame to deepface multiprocess
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
        # import pandas as pd
        import mediapipe as mp
        # tole dej v init
        # self.df = DetectFace()
        cap = cv2.VideoCapture(self.video_path)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.4
        )  # kakšen naj bo optimalen confidence tu?

        frameId = 0

        while cap.isOpened():
            ret, currentframe = cap.read()
            if not ret:
                print("Napaka pri zajemanju videa ali EOF videa")
                break
            # TODO a morem to kam drgam postavit?
            self.collect_mprocess_detection_result(currentframe, frameId)
            self.request_mprocess_detection(currentframe, frameId)

            # mediapipe variables for pose landmarks
            rgb_pose = cv2.cvtColor(currentframe, cv2.COLOR_BGR2RGB)
            results_pose = self.pose.process(rgb_pose)
            pose_landmarks = (
                results_pose.pose_landmarks.landmark
                if results_pose.pose_landmarks
                else None
            )

            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if self.trackers:
                #print("Not N-th frame, skipping detection...")
                ok, bbox = self.trackers[0].update(currentframe)
                #print(f"Tracker update result: {ok}, bbox: {bbox}")
                if ok:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(
                        currentframe, (x, y), (x + w, y + h), (255, 255, 0), 2
                    )
                    cv2.putText(
                        currentframe,
                        "Tracked between N frames",
                        (x + w, (y + 10) + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )
                else:
                    print("Tracker lost person")

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
        return []


if __name__ == "__main__":
    import cv2

    mproc.set_start_method("spawn", force=True)
    request_queue = mproc.Queue(maxsize=1)
    result_queue = mproc.Queue(maxsize=1) #al nej mam maxsize 2??
    
    ready_event = mproc.Event()
    stop_event = mproc.Event()
    
    worker = mproc.Process(
        target=process_handler,
        args=(request_queue, result_queue, stop_event, ready_event),
        daemon=True
    )
    video_files = []
    for root, dirs, files in os.walk("Data"):
        # print("Searching for files in: ", root)
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and file.lower().endswith(
                ".webm"
            ):
                #print("Found videofile: ", file)
                video_files.append(os.path.join(root, file))
    
    worker.start()
    ready_event.wait()
    csv_id = 0
    try:
        for video in video_files:
            pf = ProcessFrames(video, request_queue, result_queue, stop_event)
            df = pf.process_frames()
            
            df.to_csv(f"timeseries_{csv_id}.csv", index=False)
            print(f"Saved {len(df)} frames to timeseries.csv from video {video}")
            csv_id += 1
    finally:
        stop_event.set()
        worker.join() #al brez timeouta,al kok nej dam?
        if worker.is_alive():
            worker.terminate()
            worker.join() #a ta join je tu potreben al mam sam tega pred ifom?

    # vivit = False  # flag for vivit execution
    # # model_name = "google/vivit-b-16x2-kinetics400"
    # # model = "yolo_v8m.pt"
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
