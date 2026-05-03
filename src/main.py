import cv2
import os
import dlib
import time
import shutil
import asyncio
import threading
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

from src.detection import detector, predictor
from src.inference import eye_model, yawn_model
from src.processing import prepare_roi_square


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ALARM_PATH = os.path.join(PROJECT_ROOT, "alarm.wav")

app = FastAPI(
    title="DriveSafe-VISION-DMS-v2",
    description="High-performance Driver Monitoring System using dlib and MobileNetV2",
    version="2.0.0"
)


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
            width: min(960px, 100%);
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
            font-size: clamp(42px, 6vw, 72px);
            line-height: 0.95;
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
        }
        .submit:hover { filter: brightness(1.06); }
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
                    <h1>High-performance Driver Monitoring System</h1>
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
                <label class="drop-zone">
                    <input type="file" name="file" accept="video/*" required>
                </label>
                <button class="submit" type="submit">Analyze Now</button>
            </form>
        </section>
    </main>
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


# --- PROCESSOR CLASS ---
class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path   = video_path
        self.latest_frame = None
        self.finished     = False
        self.alert_pending = False
        self.lock         = threading.Lock()

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

        # --- SPEED OPTIMIZATION LOGIC ---
        # 1. Lower Target FPS: 7 FPS is the "sweet spot" for safety vs speed.
        # 2. Higher Skip Rate: Process fewer frames to reach the 35s goal.
        TARGET_INFERENCE_FPS = 7.0
        skip_rate = max(1, int(actual_fps / TARGET_INFERENCE_FPS))

        EYE_CLOSED_SEC       = 1.5
        YAWN_SEC             = 0.6
        EYE_FRAME_THRESHOLD  = int(TARGET_INFERENCE_FPS * EYE_CLOSED_SEC)
        YAWN_FRAME_THRESHOLD = int(TARGET_INFERENCE_FPS * YAWN_SEC)
        BLINK_IGNORE_FRAMES  = int(TARGET_INFERENCE_FPS * 0.4)

        EYE_COUNTER_CAP      = EYE_FRAME_THRESHOLD + 3
        YAWN_COUNTER_CAP     = YAWN_FRAME_THRESHOLD + 3
        ALARM_COOLDOWN_SEC   = 3.0 # Fixed cooldown for stability

        eye_frames, yawn_frames, consecutive_closed = 0, 0, 0
        last_alert = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            if frame_idx % skip_rate != 0:
                continue

            # 3. FAST DOWN-SAMPLING: Smaller frames make dlib 2-3x faster.
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Detect on the small frame
            faces = detector(gray, 0)

            status, color = "Monitoring...", (0, 255, 0)
            face_found = False

            for face in faces:
                face_found = True

                # Scale coordinates back to original frame size for dlib predictor
                scaling_factor = w / 640
                scaled_face = dlib.rectangle(
                    int(face.left() * scaling_factor), int(face.top() * scaling_factor),
                    int(face.right() * scaling_factor), int(face.bottom() * scaling_factor)
                )

                # Predict landmarks on original high-res frame for accuracy
                land = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaled_face)

                # --- EYE LOGIC ---
                closed_count = 0
                for eye_pts in [[land.part(i) for i in range(36, 42)], [land.part(i) for i in range(42, 48)]]:
                    e_roi = prepare_roi_square(frame, eye_pts, 0.55)
                    if e_roi is not None:
                        if eye_model(e_roi, training=False).numpy()[0][0] < 0.35:
                            closed_count += 1

                if closed_count == 2:
                    consecutive_closed += 1
                    if consecutive_closed > BLINK_IGNORE_FRAMES:
                        eye_frames = min(eye_frames + 1, EYE_COUNTER_CAP)
                else:
                    consecutive_closed = 0
                    eye_frames = max(0, eye_frames - 2)

                # --- YAWN LOGIC ---
                m_pts = [land.part(i) for i in range(48, 68)]
                m_roi = prepare_roi_square(frame, m_pts, 0.3)
                if m_roi is not None:
                    if yawn_model(m_roi, training=False).numpy()[0][0] > 0.85:
                        yawn_frames = min(yawn_frames + 1, YAWN_COUNTER_CAP)
                    else:
                        yawn_frames = max(0, yawn_frames - 1)

                # --- ALERT LOGIC ---
                is_drowsy = (eye_frames >= EYE_FRAME_THRESHOLD or yawn_frames >= YAWN_FRAME_THRESHOLD)
                if is_drowsy:
                    status, color = "!! DROWSY !!", (0, 0, 255)
                    if (time.time() - last_alert) > ALARM_COOLDOWN_SEC:
                        self.trigger_browser_alarm()
                        last_alert = time.time()
                        # Partial reset to prevent immediate re-beeping
                        eye_frames = EYE_FRAME_THRESHOLD // 2
                        yawn_frames = YAWN_FRAME_THRESHOLD // 2

                cv2.rectangle(frame, (scaled_face.left(), scaled_face.top()),
                              (scaled_face.right(), scaled_face.bottom()), color, 2)

            if not face_found:
                eye_frames = max(0, eye_frames - 1)
                yawn_frames = max(0, yawn_frames - 1)

            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.latest_frame = buffer.tobytes()

        cap.release()
        self.finished = True
        if os.path.exists(self.video_path):
            os.remove(self.video_path)


# --- STREAM GENERATOR ---
async def stream_frames(processor: VideoProcessor):
    stream_fps = 20
    frame_delay = 1.0 / stream_fps
    last_frame = None

    while not processor.finished or processor.get_latest_frame() is not None:
        t_start = time.time()
        frame_bytes = processor.get_latest_frame()
        if frame_bytes: last_frame = frame_bytes

        if last_frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')

        elapsed = time.time() - t_start
        sleep_time = frame_delay - elapsed
        if sleep_time > 0: await asyncio.sleep(sleep_time)


# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def main():
    return HOME_HTML


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
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
        return StreamingResponse(stream_frames(processor), media_type="multipart/x-mixed-replace; boundary=frame")
    return "No video."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
