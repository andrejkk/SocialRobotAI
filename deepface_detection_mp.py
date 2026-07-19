class DetectFace:
    def __init__(self, DeepFace):
        # from deepface import DeepFace
        self.DeepFace = DeepFace
        # print("DEEPFACE ALSO IMPORTED!!!!DKNSFNKSDFNKNKSFKS")
        pass

    def faceDetection(self, frame):
        #print("Did my worker die HERE???🥀🥀🥀🥀")
        try:
            if frame is None:
                print("Pridobljen prazen frame za detekcijo obraza.")
                return None
            # baje bi rabu upscaleat resolucijo, da bi se bolje obnašali drugi detector backendi?
            # lahko bi tudi uporabil mobilenet za realtime facial expression recognition?
            deepface_result = self.DeepFace.analyze(
                frame,
                actions=["emotion"],
                detector_backend="centerface",
                enforce_detection=True,  # prej sm meu False nastavljen skoz sam zdej debuggam
                # če imam enforce detection false, mi failed detection vrne celoten frame
                # kjer je conf 0.0, ssd mediapipe in ostali ne delujejo dobro na lowres blurry videih
                # zato enforc detection true s takim backendom ne naredi nič, tudi yunet celo ne dela pri tej resoluciji
            )
            # wrappaj v seznam v primeru če deepface_result ni seznam, ker DeepFace.analyze vrne dict za en face in list za več faceov
            # if not isinstance(deepface_result, list):
            #     deepface_result = [deepface_result]
            multiple_bboxes = []
            for detected_face in deepface_result if deepface_result is not None else []:
                fa = detected_face["region"]
                x, y, w, h = (
                    fa["x"],
                    fa["y"],
                    fa["w"],
                    fa["h"],
                )
                multiple_bboxes.append(
                    {
                        "bbox": (x, y, w, h),
                        "conf": detected_face["face_confidence"],
                        "emo": None,
                    }
                )
            return multiple_bboxes
        except Exception as e:
            print(f"Napaka pri detekciji obraza z DeepFace: {e}")
            return None


def process_handler(request_queue, result_queue, stop_event, ready_event):
    #import os, sys, traceback
    from deepface import DeepFace
    import numpy as np
    #import numpy as np
    # self.DeepFace = DeepFace
    #print("DEEPFACE ALSO IMPORTED!!!!DKNSFNKSDFNKNKSFKS")
    df = DetectFace(DeepFace)

    print(
        "🔥🔥🔥🔥HANDLER STARTED, DEEPFACE IMPORTED!!!!!!!!!!!!!---------------------------------",
        flush=True,
    )
    #sys.stdout.flush()
    print("Warming up...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    df.faceDetection(dummy)
    print("DeepFace ready")

    ready_event.set()
    
    try:
        #print("PID:", os.getpid(), flush=True)
        while not stop_event.is_set():
        # while not stop_event is set:
            #print("🔥WORKER ALIVE LOOP-----------------------------------", flush=True)
            try:
                #print("Semle sem zdaj poklical svoj detectface class 🙈🙈🙈")
                frameid, frame = request_queue.get() #no wait al kšn timeout al ne? TODO 
                det_result = df.faceDetection(frame)
                # print("🔥GOT FRAME V WORKER:", frameid, flush=True)
                # print("Dubu sm bboxse v workerju: ", det_result, "za frame: ", frameid)
                result_queue.put((frameid, det_result))

            except Exception as e:
                print("EMPTY:", e, flush=True)

    except Exception:
        print("HANDLER EXCEPTION!!!!!")
        #traceback.print_exc()


# def process_handler(request_queue, result_queue, stop_event):
#     # import sys
#     # import time
#     # time.sleep(15)
#     # print("ENTERED PROCESS_HANDLER", flush=True)
#     # sys.stdout.flush()
#     print("PREDEN SE INICIALIZIRA DETECTFACE SEM VSAJ PRIŠEL NOTRI V PROC HANDLER")
#     det = DetectFace()
#     # print("AFTER DETECTFACE INIT", flush=True)

#     # print("[process_handler] worker started", flush=True)
#     print("🔥 STEP 1 ENTERED PROCESS HANDLER", flush=True)
#     # import os
#     # print("PID:", os.getpid(), flush=True)

#     while not stop_event.is_set():
#         try:
#             print("Request queue getta frame v handler")
#             frameid, frame = request_queue.get(timeout=1) #al dam 0.5?
#         except queue.Empty:
#             print("Queue je empty handler")
#             continue
#         bboxes_result = det.faceDetection(frame)
#         try:
#             print("putta no wait handler ")
#             result_queue.put_nowait((frameid, bboxes_result or []))
#         except queue.Full:
#             try:
#                 print("queue handler full")
#                 result_queue.get_nowait()
#                 result_queue.put_nowait((frameid, bboxes_result or []))
#             except queue.Empty:
#                 print("queue empty handler")
#                 pass

# nevem a nej mam to tko tud tuki al ne? #TODO
# if __name__ == "__main__":
#     # from deepface import DeepFace
#     # print("DEEPFACE IMPORTEDDDDDSKDFDSFN")
#     process_handler()
