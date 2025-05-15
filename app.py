import time
import threading
import traceback
import json
import queue
import cv2
import numpy as np
import pyaudio
from flask import Flask, render_template, Response, stream_with_context
from tensorflow.keras.models import load_model

# INNIT MATE
app = Flask(__name__)
model = load_model("Models/face_emotion_cnn.h5")
emotion_labels = [
    "anger","contempt","disgust","fear",
    "happiness","neutrality","sadness","surprise"
]

# FACE DETECTION SETUP
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# SMOOTHING & DETECTION PARAMETERS
DETECTION_INTERVAL = 2
SMOOTHING_ALPHA = 0.9
PADDING_PERCENT = 0.2
frame_counter = 0
last_bbox = None
smoothed_bbox = None

# CAMERA SETUP
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not start camera :(")
frame_lock   = threading.Lock()
latest_frame = None

def camera_loop():
    global latest_frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.01)
    except Exception:
        traceback.print_exc()

# AUDIO CAPTURE SETUP
RATE     = 44100
CHUNK    = int(RATE * 0.1)
FORMAT   = pyaudio.paInt16
CHANNELS = 1
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)
audio_queue   = queue.Queue()
emotion_queue = queue.Queue()
start_time    = time.time()

def audio_capture_loop():
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            y    = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
            vol  = float(np.sqrt(np.mean(y**2)))
            fft      = np.fft.rfft(y)
            mags     = np.abs(fft)
            peak_idx = np.argmax(mags[1:]) + 1
            freqs    = np.fft.rfftfreq(len(y), d=1.0/RATE)
            pitch    = float(freqs[peak_idx])
            t = time.time() - start_time
            audio_queue.put({"time": t, "volume": vol, "pitch": pitch})
    except Exception:
        traceback.print_exc()

# VIDEO FRAME GENERATOR
def gen_frames():
    global frame_counter, last_bbox, smoothed_bbox
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        frame_counter += 1
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_counter % DETECTION_INTERVAL == 0 or last_bbox is None:
            faces = face_cascade.detectMultiScale(gray_full, 1.1, 5)
            if len(faces) > 0:
                last_bbox = max(faces, key=lambda b: b[2] * b[3])
        if last_bbox is None:
            continue
        if smoothed_bbox is None:
            smoothed_bbox = list(last_bbox)
        else:
            smoothed_bbox = [
                SMOOTHING_ALPHA * last_bbox[i] + (1 - SMOOTHING_ALPHA) * smoothed_bbox[i]
                for i in range(4)
            ]

        x, y, w, h = smoothed_bbox
        pad_w = w * PADDING_PERCENT
        pad_h = h * PADDING_PERCENT
        x0 = int(max(x - pad_w/2, 0))
        y0 = int(max(y - pad_h/2, 0))
        x1 = int(min(x + w + pad_w/2, frame.shape[1]))
        y1 = int(min(y + h + pad_h/2, frame.shape[0]))
        x_, y_, w_, h_ = x0, y0, x1-x0, y1-y0
        cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)
        crop    = frame[y_:y_+h_, x_:x_+w_]
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224))
        img     = resized.astype("float32") / 255.0
        img     = img[None, ..., None]
        preds = model.predict(img, verbose=0)[0]
        top   = emotion_labels[np.argmax(preds)]
        emotion_queue.put({
            "time": time.time() - start_time,
            "preds": {emo: float(p * 100) for emo, p in zip(emotion_labels, preds)},
            "top": top
        })
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        
# GRAYSCALE INPUT VIEW
def gen_input_view():
    global smoothed_bbox
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None or smoothed_bbox is None:
            time.sleep(0.01)
            continue
        x, y, w, h = smoothed_bbox
        pad_w = w * PADDING_PERCENT
        pad_h = h * PADDING_PERCENT
        x0 = int(max(x - pad_w/2, 0))
        y0 = int(max(y - pad_h/2, 0))
        x1 = int(min(x + w + pad_w/2, frame.shape[1]))
        y1 = int(min(y + h + pad_h/2, frame.shape[0]))
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        inp  = cv2.resize(gray, (224, 224))
        ret, buf = cv2.imencode('.jpg', inp)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        
# ROUTES
@app.route('/')
def index():
    global start_time
    start_time = time.time()
    with audio_queue.mutex:
        audio_queue.queue.clear()
    with emotion_queue.mutex:
        emotion_queue.queue.clear()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/input_feed')
def input_feed():
    return Response(
        gen_input_view(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/audio_data')
def audio_data():
    def event_stream():
        try:
            while True:
                data = audio_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except Exception:
            traceback.print_exc()
    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream'
    )

@app.route('/emotion_data')
def emotion_data():
    def event_stream():
        try:
            while True:
                data = emotion_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except Exception:
            traceback.print_exc()
    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream'
    )

# STARTUP
if __name__ == '__main__':
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=audio_capture_loop, daemon=True).start()
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True,
        use_reloader=False
    )
    cap.release()
    stream.stop_stream()
    stream.close()
    p.terminate()
