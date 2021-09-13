# source : https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

# from pyimagesearch.motion_detection import SingleMotionDetector


from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import time
import cv2
from emotiondetection import EmotionDetector
from PIL import Image
import copy
import numpy as np

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
emotion_pred_score = []
em_ans_scores = []

# initialize a flask object
app = Flask(__name__)
# app = Flask(__name__, static_url_path="/static", static_folder="app/static")
# initialize the video stream and allow the camera sensor to

# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2.0)

modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"

emotions = {
    0: "Happy",
    1: "Angry",
    2: "Disgust",
    3: "Fear",
    4: "Sad",
    5: "Contempt",
    6: "Surprise",
}
global net
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


@app.route("/")
def index():
    return render_template("index.html")


def detect_emotion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock, emotion_pred_score
    # initialize the motion detector and the total number of frames
    # read thus far

    ed = EmotionDetector(frameCount)

    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        ret, frame = vs.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ##############
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame_rgb, 1.0, (224, 224), [104, 117, 123], False, False
        )

        net.setInput(blob)
        detections = net.forward()

        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                if (224 < h) & (224 < w):

                    # draw box over face
                    frame = cv2.rectangle(
                        frame,
                        (x1 - 10, y1 - 10),
                        (x1 + 214, y1 + 214),
                        (0, 255, 0),
                        2,
                    )

                    crop_img = frame[y1 - 10 : y1 + 214, x1 - 10 : x1 + 214]

                    (ch, cw) = crop_img.shape[:2]

                    if (ch == 224) & (cw == 224):
                        total += 1

                        img = Image.fromarray(crop_img)
                        ed.update(img, total)

                        if total >= frameCount:
                            emotion_pred_score = ed.detect()
                            em_ans_scores = [
                                emotion_pred_score[1],
                                emotion_pred_score[3],
                                emotion_pred_score[5],
                                emotion_pred_score[7],
                                emotion_pred_score[9],
                                emotion_pred_score[11],
                                emotion_pred_score[13],
                            ]
                            em_ans = np.argmax(em_ans_scores)

                            cv2.putText(
                                frame,
                                emotions[em_ans],
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (36, 255, 12),
                                2,
                            )

                    else:
                        print("Your face is not properly aligned")

                else:
                    print("Take away your face from camera")
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream

    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == "__main__":

    frame_count = 5

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_emotion, args=(frame_count,))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(debug=True, threaded=True, use_reloader=False)
# release the video stream pointer
vs.release()