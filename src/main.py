import cv2
import numpy as np
import os
import dlib
import time
import shutil
import asyncio
import threading

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
try:
    from pygame import mixer
except Exception:
    mixer = None

from src.detection import detector, predictor, clamp_rect
from src.inference import eye_model, yawn_model
from src.processing import prepare_roi_square


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ALARM_PATH = os.environ.get("ALARM_PATH", os.path.join(PROJECT_ROOT, "alarm.wav"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(PROJECT_ROOT, "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


beep_sound = None
if mixer is not None:
    try:
        mixer.init()
        beep_sound = mixer.Sound(ALARM_PATH)
    except Exception:
        beep_sound = None

app = FastAPI()

HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DriveSafe-VISION-DMS-v2</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #080b0f;
            --panel: #111820;
            --panel-2: #17212b;
            --text: #f5f7fb;
            --muted: #9da9b7;
            --accent: #ff3b30;
            --accent-2: #32d583;
            --line: rgba(255, 255, 255, 0.1);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 20% 20%, rgba(255, 59, 48, 0.12), transparent 28%),
                radial-gradient(circle at 80% 10%, rgba(50, 213, 131, 0.1), transparent 24%),
                linear-gradient(135deg, #070a0d 0%, #101820 55%, #080b0f 100%);
        }
        .page {
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 32px;
        }
        .shell {
            width: min(1040px, 100%);
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 28px;
            align-items: stretch;
        }
        .brand, .upload-panel {
            border: 1px solid var(--line);
            background: rgba(17, 24, 32, 0.82);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(16px);
            border-radius: 8px;
        }
        .brand {
            padding: 42px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 460px;
        }
        .eyebrow {
            color: var(--accent-2);
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        h1 {
            margin: 18px 0 12px;
            font-size: clamp(40px, 5vw, 68px);
            line-height: 1;
            letter-spacing: 0;
        }
        .summary {
            max-width: 560px;
            color: var(--muted);
            font-size: 17px;
            line-height: 1.65;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 32px;
        }
        .metric {
            padding: 16px;
            background: rgba(255,255,255,0.045);
            border: 1px solid var(--line);
            border-radius: 8px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric:hover {
            transform: translateY(-3px);
            border-color: rgba(50, 213, 131, 0.35);
        }
        .metric strong { display: block; font-size: 22px; }
        .metric span { color: var(--muted); font-size: 12px; }
        .upload-panel {
            padding: 34px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 22px;
        }
        .upload-title {
            margin: 0;
            font-size: 26px;
            line-height: 1.2;
        }
        .drop-zone {
            display: grid;
            gap: 14px;
            padding: 28px;
            border: 1px dashed rgba(255,255,255,0.24);
            border-radius: 8px;
            background: rgba(255,255,255,0.04);
            transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
        }
        .drop-zone.dragging {
            border-color: var(--accent-2);
            background: rgba(50, 213, 131, 0.08);
            transform: translateY(-2px);
        }
        .file-name {
            min-height: 20px;
            color: var(--accent-2);
            font-size: 14px;
            font-weight: 700;
        }
        input[type="file"] {
            width: 100%;
            color: var(--muted);
        }
        input[type="file"]::file-selector-button, button {
            border: 0;
            border-radius: 8px;
            padding: 13px 18px;
            font-weight: 800;
            cursor: pointer;
        }
        input[type="file"]::file-selector-button {
            margin-right: 14px;
            color: #101820;
            background: #f5f7fb;
        }
        .submit {
            width: 100%;
            color: white;
            background: linear-gradient(135deg, #ff3b30, #d91f2a);
            box-shadow: 0 14px 34px rgba(255, 59, 48, 0.28);
            font-size: 16px;
            transition: transform 0.2s ease, filter 0.2s ease;
        }
        .submit:hover { filter: brightness(1.06); transform: translateY(-2px); }
        .hint {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.5;
        }
        @media (max-width: 820px) {
            .shell { grid-template-columns: 1fr; }
            .brand { min-height: 360px; padding: 30px; }
            .metrics { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="shell">
            <div class="brand">
                <div>
                    <div class="eyebrow">High-performance Driver Monitoring System</div>
                    <h1>DriveSafe-VISION DMS v2</h1>
                    <p class="summary">Upload a driving video and monitor fatigue signals with eye-closure and yawn detection.</p>
                </div>
                <div class="metrics" aria-label="monitoring signals">
                    <div class="metric"><strong>Eyes</strong><span>closure tracking</span></div>
                    <div class="metric"><strong>Yawn</strong><span>mouth movement</span></div>
                    <div class="metric"><strong>Alert</strong><span>browser alarm</span></div>
                </div>
            </div>
            <form class="upload-panel" action="/upload" method="post" enctype="multipart/form-data">
                <div>
                    <h2 class="upload-title">Analyze video</h2>
                    <p class="hint">Choose a video file to start the drowsiness check.</p>
                </div>
                <label id="dropZone" class="drop-zone">
                    <input id="videoInput" type="file" name="file" accept="video/*" required>
                    <span id="fileName" class="file-name">No video selected</span>
                </label>
                <button id="submitButton" class="submit" type="submit">Analyze Now</button>
            </form>
        </section>
    </main>
    <script>
        const dropZone = document.getElementById('dropZone');
        const videoInput = document.getElementById('videoInput');
        const fileName = document.getElementById('fileName');
        const submitButton = document.getElementById('submitButton');

        ['dragenter', 'dragover'].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropZone.classList.add('dragging');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropZone.classList.remove('dragging');
            });
        });

        dropZone.addEventListener('drop', (event) => {
            const files = event.dataTransfer.files;
            if (files.length) {
                videoInput.files = files;
                fileName.textContent = files[0].name;
            }
        });

        videoInput.addEventListener('change', () => {
            fileName.textContent = videoInput.files.length ? videoInput.files[0].name : 'No video selected';
        });

        document.querySelector('form').addEventListener('submit', () => {
            submitButton.textContent = 'Uploading...';
            submitButton.disabled = true;
        });
    </script>
</body>
</html>
"""


PROCESSING_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>High-performance Driver Monitoring System</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #080b0f;
            --panel: #111820;
            --text: #f5f7fb;
            --muted: #9da9b7;
            --accent: #ff3b30;
            --ok: #32d583;
            --line: rgba(255, 255, 255, 0.1);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background: linear-gradient(135deg, #070a0d 0%, #111923 55%, #080b0f 100%);
        }
        .page {
            min-height: 100vh;
            padding: 28px;
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 20px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 18px 22px;
            border: 1px solid var(--line);
            background: rgba(17, 24, 32, 0.86);
            border-radius: 8px;
        }
        h1 { margin: 0; font-size: 26px; letter-spacing: 0; }
        .status-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .badge {
            min-width: 116px;
            text-align: center;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(50, 213, 131, 0.14);
            color: var(--ok);
            font-weight: 800;
            border: 1px solid rgba(50, 213, 131, 0.32);
        }
        .badge.alert {
            background: rgba(255, 59, 48, 0.16);
            color: #ffb4ae;
            border-color: rgba(255, 59, 48, 0.38);
        }
        button {
            border: 0;
            border-radius: 8px;
            padding: 11px 16px;
            font-weight: 800;
            cursor: pointer;
            color: #101820;
            background: #f5f7fb;
        }
        button.enabled {
            color: #03150c;
            background: var(--ok);
        }
        .viewer {
            min-height: 0;
            display: grid;
            place-items: center;
            border: 1px solid var(--line);
            background:
                radial-gradient(circle at 50% 0%, rgba(255, 59, 48, 0.12), transparent 32%),
                rgba(17, 24, 32, 0.72);
            border-radius: 8px;
            overflow: hidden;
            padding: 18px;
        }
        .stream {
            width: min(1180px, 100%);
            max-height: calc(100vh - 160px);
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.14);
            background: #050607;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.42);
        }
        @media (max-width: 720px) {
            .page { padding: 14px; }
            header { align-items: flex-start; flex-direction: column; }
            h1 { font-size: 22px; }
            .status-row { width: 100%; }
            button, .badge { flex: 1; }
        }
    </style>
</head>
<body>
    <main class="page">
        <header>
            <h1>High-performance Driver Monitoring System Analysis</h1>
            <div class="status-row">
                <div id="statusBadge" class="badge">Monitoring</div>
                <button id="enableAlarm" type="button">Enable Alarm</button>
            </div>
        </header>
        <section class="viewer">
            <img class="stream" src="/stream_result" alt="Processed video stream">
        </section>
    </main>
    <audio id="alarmAudio" src="/alarm.wav" preload="auto"></audio>
    <script>
        const alarm = document.getElementById('alarmAudio');
        const enableButton = document.getElementById('enableAlarm');
        const badge = document.getElementById('statusBadge');
        let alarmEnabled = false;

        enableButton.addEventListener('click', async () => {
            try {
                alarm.volume = 0;
                await alarm.play();
                alarm.pause();
                alarm.currentTime = 0;
                alarm.volume = 1;
                alarmEnabled = true;
                enableButton.textContent = 'Alarm Enabled';
                enableButton.classList.add('enabled');
            } catch (error) {
                alarmEnabled = false;
                enableButton.textContent = 'Enable Alarm';
            }
        });

        async function pollAlarm() {
            try {
                const response = await fetch('/alarm_status', { cache: 'no-store' });
                const data = await response.json();

                if (data.alert) {
                    badge.textContent = 'Drowsy';
                    badge.classList.add('alert');
                    if (alarmEnabled) {
                        alarm.currentTime = 0;
                        alarm.play().catch(() => {});
                    }
                    setTimeout(() => {
                        badge.textContent = 'Monitoring';
                        badge.classList.remove('alert');
                    }, 2200);
                }

                if (data.finished) {
                    badge.textContent = 'Complete';
                }
            } catch (error) {}
        }

        setInterval(pollAlarm, 650);
    </script>
</body>
</html>
"""

def play_alarm():
    if mixer is None or beep_sound is None:
        return
    try:
        if not mixer.get_busy():
            beep_sound.play()
    except Exception:
        pass


# =========================
# PROCESSOR
# =========================
class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.latest_frame = None
        self.finished = False
        self.alert_pending = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._process, daemon=True).start()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def trigger_browser_alarm(self):
        with self.lock:
            self.alert_pending = True

    def consume_alert(self):
        with self.lock:
            alert = self.alert_pending
            self.alert_pending = False
            return alert

    def _process(self):
        cap = cv2.VideoCapture(self.video_path)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = 30.0

        # =========================
        # REAL-TIME SPEED SETTINGS
        # =========================
        TARGET_INFERENCE_FPS = 5.0
        skip_rate = max(1, int(round(actual_fps / TARGET_INFERENCE_FPS)))

        DETECT_W, DETECT_H = 432, 243
        LANDMARK_REFRESH_EVERY = 3
        FACE_DETECT_REFRESH_EVERY = 4
        YAWN_EVAL_EVERY = 2

        # =========================
        # DETECTION SETTINGS
        # =========================
        EYE_CLOSED_THRESHOLD = 0.35
        EYE_CLOSED_SEC = 0.85
        OPEN_GRACE_SEC = 0.18

        YAWN_THRESHOLD = 0.85
        YAWN_SEC = 0.60
        YAWN_FRAME_THRESHOLD = max(1, int(TARGET_INFERENCE_FPS * YAWN_SEC))
        YAWN_COUNTER_CAP = YAWN_FRAME_THRESHOLD + 2

        ALARM_COOLDOWN_SEC = 3.0

        frame_idx = 0
        processed_idx = 0

        eye_closed_start_time = None
        eye_open_grace_start = None
        yawn_frames = 0
        last_alert = 0.0

        cached_face = None
        cached_land = None
        last_yawn_pred = None
        last_avg_eye_pred = None
        last_eye_closed_now = False
        last_eye_closed_duration = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip_rate != 0:
                continue

            processed_idx += 1
            h, w = frame.shape[:2]

            status = "Monitoring..."
            color = (0, 255, 0)
            face_found = False

            # -------------------------
            # FACE DETECTION (NOT EVERY FRAME)
            # -------------------------
            need_face_refresh = (
                cached_face is None or
                processed_idx % FACE_DETECT_REFRESH_EVERY == 1
            )

            if need_face_refresh:
                small_frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
                small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = detector(small_gray, 0)

                if faces:
                    face = faces[0]
                    sx = w / float(DETECT_W)
                    sy = h / float(DETECT_H)

                    scaled_face = dlib.rectangle(
                        int(face.left() * sx),
                        int(face.top() * sy),
                        int(face.right() * sx),
                        int(face.bottom() * sy)
                    )
                    cached_face = clamp_rect(scaled_face, w, h)
                else:
                    cached_face = None
                    cached_land = None

            if cached_face is not None:
                face_found = True

                # -------------------------
                # LANDMARKS (NOT EVERY FRAME)
                # -------------------------
                need_landmark_refresh = (
                    cached_land is None or
                    processed_idx % LANDMARK_REFRESH_EVERY == 1
                )

                if need_landmark_refresh:
                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cached_land = predictor(gray_full, cached_face)

                land = cached_land

                left_eye_pts = [land.part(i) for i in range(36, 42)]
                right_eye_pts = [land.part(i) for i in range(42, 48)]
                mouth_pts = [land.part(i) for i in range(48, 68)]

                # -------------------------
                # EYE MODEL (BATCHED)
                # -------------------------
                left_roi = prepare_roi_square(frame, left_eye_pts, pad=0.42, out_size=224)
                right_roi = prepare_roi_square(frame, right_eye_pts, pad=0.42, out_size=224)

                left_eye_pred = None
                right_eye_pred = None
                avg_eye_pred = last_avg_eye_pred

                eye_batch = []
                eye_tags = []

                if left_roi is not None:
                    eye_batch.append(left_roi)
                    eye_tags.append("left")
                if right_roi is not None:
                    eye_batch.append(right_roi)
                    eye_tags.append("right")

                if eye_batch:
                    eye_batch_np = np.stack(eye_batch, axis=0)
                    preds = eye_model(eye_batch_np, training=False).numpy().reshape(-1)

                    for tag, pred in zip(eye_tags, preds):
                        if tag == "left":
                            left_eye_pred = float(pred)
                        elif tag == "right":
                            right_eye_pred = float(pred)

                    valid_eye_preds = [p for p in [left_eye_pred, right_eye_pred] if p is not None]
                    if valid_eye_preds:
                        avg_eye_pred = float(sum(valid_eye_preds) / len(valid_eye_preds))
                        last_avg_eye_pred = avg_eye_pred

                both_closed = (
                    left_eye_pred is not None and
                    right_eye_pred is not None and
                    left_eye_pred < EYE_CLOSED_THRESHOLD and
                    right_eye_pred < EYE_CLOSED_THRESHOLD
                )

                avg_closed = (
                    avg_eye_pred is not None and
                    avg_eye_pred < (EYE_CLOSED_THRESHOLD + 0.02)
                )

                eye_closed_now = both_closed or avg_closed
                now = time.time()

                if eye_closed_now:
                    eye_open_grace_start = None
                    if eye_closed_start_time is None:
                        eye_closed_start_time = now
                    eye_closed_duration = now - eye_closed_start_time
                else:
                    if eye_closed_start_time is not None:
                        if eye_open_grace_start is None:
                            eye_open_grace_start = now
                        elif (now - eye_open_grace_start) > OPEN_GRACE_SEC:
                            eye_closed_start_time = None
                            eye_open_grace_start = None

                    if eye_closed_start_time is not None:
                        eye_closed_duration = now - eye_closed_start_time
                    else:
                        eye_closed_duration = 0.0

                last_eye_closed_now = eye_closed_now
                last_eye_closed_duration = eye_closed_duration
                eye_drowsy = eye_closed_duration >= EYE_CLOSED_SEC

                # -------------------------
                # YAWN MODEL (LESS OFTEN)
                # -------------------------
                if processed_idx % YAWN_EVAL_EVERY == 0:
                    mouth_roi = prepare_roi_square(frame, mouth_pts, pad=0.25, out_size=224)
                    if mouth_roi is not None:
                        mouth_batch = np.expand_dims(mouth_roi, axis=0)
                        last_yawn_pred = float(
                            yawn_model(mouth_batch, training=False).numpy()[0][0]
                        )

                        if last_yawn_pred > YAWN_THRESHOLD:
                            yawn_frames = min(yawn_frames + 1, YAWN_COUNTER_CAP)
                        else:
                            yawn_frames = max(0, yawn_frames - 1)

                yawn_drowsy = yawn_frames >= YAWN_FRAME_THRESHOLD

                # -------------------------
                # ALERT LOGIC
                # -------------------------
                is_drowsy = eye_drowsy or yawn_drowsy

                if is_drowsy:
                    status = "!! DROWSY !!"
                    color = (0, 0, 255)

                    if (time.time() - last_alert) > ALARM_COOLDOWN_SEC:
                        threading.Thread(target=play_alarm, daemon=True).start()
                        self.trigger_browser_alarm()
                        last_alert = time.time()

                    if eye_drowsy:
                        eye_closed_start_time = time.time() - 0.30

                    if yawn_drowsy:
                        yawn_frames = max(0, YAWN_FRAME_THRESHOLD // 2)

                elif last_eye_closed_now:
                    status = "Eyes Closed..."
                    color = (0, 165, 255)

                cv2.rectangle(
                    frame,
                    (cached_face.left(), cached_face.top()),
                    (cached_face.right(), cached_face.bottom()),
                    color,
                    2
                )

                # Minimal overlay only
                cv2.putText(
                    frame,
                    status,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2
                )
                if last_avg_eye_pred is not None:
                    cv2.putText(
                        frame,
                        f"Eye:{last_avg_eye_pred:.3f} Time:{last_eye_closed_duration:.2f}/{EYE_CLOSED_SEC:.2f}",
                        (30, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 255),
                        2
                    )

            else:
                eye_closed_start_time = None
                eye_open_grace_start = None
                yawn_frames = max(0, yawn_frames - 1)
                cached_land = None
                last_avg_eye_pred = None
                last_eye_closed_now = False
                last_eye_closed_duration = 0.0

                cv2.putText(
                    frame,
                    status,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2
                )

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self.lock:
                self.latest_frame = buffer.tobytes()

        cap.release()
        self.finished = True

        if os.path.exists(self.video_path):
            os.remove(self.video_path)


# =========================
# STREAM GENERATOR
# =========================
async def stream_frames(processor: VideoProcessor):
    stream_fps = 20
    frame_delay = 1.0 / stream_fps
    last_frame = None

    while not processor.finished or processor.get_latest_frame() is not None:
        t_start = time.time()

        frame_bytes = processor.get_latest_frame()
        if frame_bytes:
            last_frame = frame_bytes

        if last_frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                last_frame +
                b'\r\n'
            )

        elapsed = time.time() - t_start
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def main():
    return HOME_HTML


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processor = VideoProcessor(temp_path)
    processor.start()
    app.state.processor = processor

    return HTMLResponse(content=PROCESSING_HTML)


@app.get("/alarm.wav")
async def alarm_sound():
    return FileResponse(ALARM_PATH, media_type="audio/wav", filename="alarm.wav")


@app.get("/alarm_status")
async def alarm_status():
    processor = getattr(app.state, "processor", None)
    if processor:
        return {"alert": processor.consume_alert(), "finished": processor.finished}
    return {"alert": False, "finished": False}

@app.get("/stream_result")
async def stream_result():
    processor = getattr(app.state, "processor", None)
    if processor:
        return StreamingResponse(
            stream_frames(processor),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    return HTMLResponse("No video.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


