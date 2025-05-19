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
import librosa


# INNIT MATE
app = Flask(__name__)
model = load_model("Models/face_emotion_cnn.h5")
audio_model = load_model("Models/audio_emotion_model.h5")
audio_emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


emotion_labels = [
    "anger","contempt","disgust","fear",
    "happiness","neutrality","sadness","surprise"
]

# CAMERA STUFF
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
        print("Camera thread exception oh no")
        traceback.print_exc()

# AUDIO STUFF
RATE     = 44100
CHUNK    = int(RATE * 0.1)
FORMAT   = pyaudio.paInt16
CHANNELS = 1

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

audio_queue = queue.Queue()
mfcc_buffer = []
start_time  = time.time()

def audio_capture_loop():
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            y    = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

            vol = float(np.sqrt(np.mean(y**2)))

            fft  = np.fft.rfft(y)
            mags = np.abs(fft)
            peak_idx = np.argmax(mags[1:]) + 1
            freqs    = np.fft.rfftfreq(len(y), d=1.0/RATE)
            pitch    = float(freqs[peak_idx])

            t = time.time() - start_time


            y = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
            mfcc_buffer.append(y)

            # Keep last 2 seconds (e.g. 20 chunks of 0.1s)
            if len(mfcc_buffer) > 20:
                mfcc_buffer.pop(0)

            # Once buffer is full, run inference
            if len(mfcc_buffer) == 20:
                full_audio = np.concatenate(mfcc_buffer)
                mfcc = librosa.feature.mfcc(y=full_audio, sr=RATE, n_mfcc=40)
                if mfcc.shape[1] < 300:
                    mfcc = np.pad(mfcc, ((0,0), (0, 300 - mfcc.shape[1])), mode='constant')
                else:
                    mfcc = mfcc[:, :300]
                mfcc = mfcc.T[np.newaxis, :, :]  # (1, time, features)

                preds = audio_model.predict(mfcc, verbose=0)[0]
                top_idx = np.argmax(preds)
                audio_emotion = audio_emotion_labels[top_idx]
                audio_queue.put({"time": t,
                                "volume": vol,
                                "pitch": pitch, 
                                "emotion": audio_emotion, 
                                "emotion_confidence": float(preds[top_idx])})
            else:
                audio_queue.put({
                    "time": t,
                    "pitch": pitch,
                    "emotion": "loading",
                    "emotion_confidence": 0.0
                })
                

    except Exception:
        print("Audio thread exception bzzt")
        traceback.print_exc()

# VIDEO STUFF
def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        H, W = frame.shape[:2]
        x0, y0 = W//3, H//3
        crop   = frame[y0:y0+H//3, x0:x0+W//3]

        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224,224))
        img     = (resized.astype("float32") / 255.0)[None,...,None]

        preds = model.predict(img, verbose=0)[0]
        top   = emotion_labels[np.argmax(preds)]

        line_h = 20; margin = 10
        y_base = y0 + (H//3) - margin - len(emotion_labels)*line_h
        for i,(emo,p) in enumerate(zip(emotion_labels,preds)):
            text  = f"{emo}: {p*100:.1f}%"
            color = (0,255,0) if emo==top else (0,0,255)
            y_pos = y_base + i*line_h
            cv2.putText(frame, text, (10,y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1,cv2.LINE_AA)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret: continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

def gen_input_view():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        H, W = frame.shape[:2]
        x0, y0 = W//3, H//3
        crop = frame[y0:y0+H//3, x0:x0+W//3]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        inp  = cv2.resize(gray, (224,224))

        ret, buf = cv2.imencode('.jpg', inp)
        if not ret: continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

# ROUTES
@app.route('/')
def index():
    global start_time
    start_time = time.time()
    with audio_queue.mutex:
        audio_queue.queue.clear()
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
            print("Server event thread exception boooo")
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

    # cleanup
    cap.release()
    stream.stop_stream()
    stream.close()
    p.terminate()
