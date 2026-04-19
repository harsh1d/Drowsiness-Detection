"""
Real-Time Drowsiness Detection System — Enhanced Precision
=========================================================
Precision improvements over baseline:
  • Personal EAR/MAR calibration   – per-user adaptive thresholds
  • Rolling-average smoothing      – 5-frame window, eliminates flicker noise
  • PERCLOS metric                 – % eye-closed frames in a sliding window
  • Blink-rate monitoring          – alert on abnormally low blinks/min
  • Head-nodding detection         – tracks nose-tip drift toward chin
  • Audio alerts (winsound)        – beep with cooldown; no extra dependencies
  • Session CSV logging            – timestamped event log
  • Live FPS counter               – shows actual processing rate
  • Colour-coded HUD               – all metrics visible at a glance
"""

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# winsound is Windows-only; degrade gracefully on other platforms
try:
    import winsound
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False


# ==================== CONFIGURATION ====================

# --- Calibration ---
CALIBRATION_FRAMES    = 90    # ~3 sec at 30 fps – keep eyes open during this phase
EAR_CALIB_FACTOR      = 0.75  # personal threshold = open_eye_EAR × this factor
MAR_CALIB_FACTOR      = 1.55  # personal threshold = neutral_MAR × this factor
EAR_THRESHOLD_MIN     = 0.17  # hard lower bound after calibration
EAR_THRESHOLD_MAX     = 0.30  # hard upper bound after calibration
MAR_THRESHOLD_MIN     = 0.55
MAR_THRESHOLD_MAX     = 0.90

# --- Fall-back defaults (used before / if calibration fails) ---
EAR_THRESHOLD_DEFAULT = 0.25
MAR_THRESHOLD_DEFAULT = 0.75

# --- Eye-closure streak ---
EAR_CONSEC_FRAMES = 20   # consecutive frames below EAR threshold → alert

# --- PERCLOS ---
PERCLOS_WINDOW          = 300   # sliding window length in frames (~10 sec at 30 fps)
PERCLOS_ALERT_THRESHOLD = 0.20  # alert if eye-closed proportion exceeds 20 %

# --- Blink rate ---
BLINK_RATE_WINDOW   = 60.0  # seconds over which to count blinks
BLINK_MIN_PER_MIN   = 8     # alert if blinks/min falls below this after warm-up
BLINK_WARMUP_SEC    = 60.0  # wait this long before issuing a low-blink-rate alert

# --- Head nodding ---
HEAD_NOD_BASELINE_FRAMES = 30    # frames to establish nose-tip baseline
HEAD_NOD_THRESHOLD       = 0.12  # normalised nose-tip drop to flag a nod
HEAD_NOD_CONSEC          = 15    # consecutive frames with nod detected → alert

# --- Smoothing ---
SMOOTHING_WINDOW = 5  # frames for EAR/MAR rolling average

# --- Alerts ---
ALERT_COOLDOWN_SEC = 3.0  # minimum seconds between repeated audio beeps

# --- Paths ---
MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"
LOG_FILE   = "drowsiness_log.csv"


# ==================== UTILITY CLASSES ====================

class RollingAverage:
    """Running mean over a fixed-size window."""

    def __init__(self, window: int):
        self._buf = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def get(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0


class PerclosTracker:
    """PERCLOS – percentage of eye-closed frames in a sliding window."""

    def __init__(self, window: int):
        self._window = deque(maxlen=window)

    def update(self, ear: float, threshold: float) -> float:
        self._window.append(1 if ear < threshold else 0)
        if len(self._window) < 10:
            return 0.0
        return float(sum(self._window) / len(self._window))


class BlinkCounter:
    """
    Detects complete blink events: EAR dips below threshold then recovers.
    Short closures (≤ 10 frames) are counted as blinks.
    Longer closures are treated as drowsy eye-closure (not counted as blinks).
    """

    def __init__(self):
        self._blink_times: deque = deque()   # timestamps of recent blinks
        self._below: bool = False
        self._below_count: int = 0
        self.total: int = 0

    def update(self, ear: float, threshold: float) -> int:
        """Update state and return blinks in the last 60 seconds."""
        if ear < threshold:
            self._below = True
            self._below_count += 1
        elif self._below:
            if self._below_count <= 10:          # short closure = blink
                self._blink_times.append(time.time())
                self.total += 1
            self._below = False
            self._below_count = 0

        # Drop records older than 60 s
        cutoff = time.time() - BLINK_RATE_WINDOW
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        return len(self._blink_times)


class HeadNodDetector:
    """
    Detects forward head nodding by tracking the normalised vertical position
    of nose-tip landmark (index 30) relative to the face bounding box.
    """

    def __init__(self, baseline_frames: int = HEAD_NOD_BASELINE_FRAMES,
                 threshold: float = HEAD_NOD_THRESHOLD,
                 consec: int = HEAD_NOD_CONSEC):
        self._bf      = baseline_frames
        self._thresh  = threshold
        self._consec  = consec
        self._buf: list = []
        self._baseline: float | None = None
        self._nod_count: int = 0

    def update(self, shape: np.ndarray, face_rect) -> bool:
        """Return True when a sustained nod is detected."""
        face_h = face_rect.bottom() - face_rect.top()
        if face_h == 0:
            return False

        nose_y = shape[30][1]
        norm_y = (nose_y - face_rect.top()) / face_h

        if self._baseline is None:
            self._buf.append(norm_y)
            if len(self._buf) >= self._bf:
                self._baseline = float(np.mean(self._buf))
            return False

        if norm_y - self._baseline > self._thresh:
            self._nod_count += 1
        else:
            self._nod_count = 0

        return self._nod_count >= self._consec

    def reset_baseline(self):
        self._buf.clear()
        self._baseline = None
        self._nod_count = 0


class AlertManager:
    """Handles audio beeps with a cooldown to avoid alarm fatigue."""

    def __init__(self, cooldown: float = ALERT_COOLDOWN_SEC):
        self._cooldown = cooldown
        self._last_time: float = 0.0
        self._thread: threading.Thread | None = None

    def trigger(self) -> bool:
        """Fire audio alert if cooldown has elapsed. Returns True if fired."""
        now = time.time()
        if now - self._last_time < self._cooldown:
            return False
        self._last_time = now
        if _AUDIO_AVAILABLE:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._beep, daemon=True)
                self._thread.start()
        return True

    @staticmethod
    def _beep():
        try:
            for _ in range(3):
                winsound.Beep(1000, 300)
                time.sleep(0.1)
        except Exception:
            pass


class SessionLogger:
    """Appends drowsiness events to a CSV log file."""

    _HEADER = ["timestamp", "event", "ear", "mar", "perclos_pct", "blinks_per_min"]

    def __init__(self, filepath: str):
        self._path = filepath
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(self._HEADER)

    def log(self, event: str, ear: float, mar: float,
            perclos: float, blinks_per_min: int):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, event, f"{ear:.3f}", f"{mar:.3f}",
                f"{perclos*100:.1f}", blinks_per_min,
            ])


# ==================== DRAWING ====================

def draw_hud(frame: np.ndarray, ear: float, mar: float,
             perclos: float, blinks_per_min: int, fps: float,
             ear_thr: float, mar_thr: float,
             alert_active: bool, alert_reasons: list[str],
             calibrating: bool, calib_progress: float):
    """Render the semi-transparent HUD and alert banner."""
    h, w = frame.shape[:2]

    if calibrating:
        # Calibration progress bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        pct = int(calib_progress * 100)
        cv2.putText(frame, f"Calibrating – keep eyes open  {pct}%",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        bar_x2 = 12 + int((w - 24) * calib_progress)
        cv2.rectangle(frame, (12, 34), (w - 12, 48), (40, 40, 40), -1)
        cv2.rectangle(frame, (12, 34), (bar_x2, 48), (0, 215, 255), -1)
        return

    # Semi-transparent metric panel (top-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (265, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def _put(text, y, color):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _put(f"EAR:      {ear:.3f}  (thr {ear_thr:.2f})",
         26, (0, 80, 255) if ear < ear_thr else (0, 230, 0))
    _put(f"MAR:      {mar:.3f}  (thr {mar_thr:.2f})",
         54, (0, 165, 255) if mar > mar_thr else (0, 230, 0))
    _put(f"PERCLOS:  {perclos*100:.1f}%",
         82, (0, 0, 255) if perclos >= PERCLOS_ALERT_THRESHOLD else (0, 230, 0))
    _put(f"Blinks/min: {blinks_per_min}",
         110, (0, 165, 255) if blinks_per_min < BLINK_MIN_PER_MIN else (0, 230, 0))
    _put(f"FPS: {fps:.1f}",
         138, (180, 180, 180))

    # Alert banner (bottom)
    if alert_active:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 16)
        cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "!!! DROWSINESS ALERT !!!",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "  |  ".join(alert_reasons),
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "AWAKE", (w - 110, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)


# ==================== HELPER FUNCTIONS ====================

def calculate_eye_aspect_ratio(eye: np.ndarray) -> float:
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h  = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h)


def calculate_mouth_aspect_ratio(mouth: np.ndarray) -> float:
    v1 = dist.euclidean(mouth[2], mouth[10])
    v2 = dist.euclidean(mouth[4], mouth[8])
    v3 = dist.euclidean(mouth[3], mouth[9])
    h  = dist.euclidean(mouth[0], mouth[6])
    return (v1 + v2 + v3) / (2.0 * h)


# ==================== MAIN ====================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'")
        print("Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    print("Initialising drowsiness detection system (enhanced precision)…")

    face_detector      = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(MODEL_PATH)

    (le_s, le_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (re_s, re_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (m_s,  m_e)  = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ---- Precision components ----
    ear_avg    = RollingAverage(SMOOTHING_WINDOW)
    mar_avg    = RollingAverage(SMOOTHING_WINDOW)
    perclos    = PerclosTracker(PERCLOS_WINDOW)
    blinker    = BlinkCounter()
    nod_det    = HeadNodDetector()
    alert_mgr  = AlertManager()
    logger     = SessionLogger(LOG_FILE)

    # ---- Adaptive thresholds (set during calibration) ----
    ear_thr = EAR_THRESHOLD_DEFAULT
    mar_thr = MAR_THRESHOLD_DEFAULT

    # ---- Calibration state ----
    calibrating      = True
    calib_ear_buf: list[float] = []
    calib_mar_buf: list[float] = []

    # ---- Runtime state ----
    eye_closed_frames = 0
    session_start     = time.time()

    # ---- FPS tracking ----
    fps_counter = 0
    fps_timer   = time.time()
    fps         = 0.0

    print(f"Starting calibration ({CALIBRATION_FRAMES} frames).")
    print("Keep your eyes open and look at the camera.")
    print("Controls:  q = quit   r = recalibrate")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps         = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)

        # Working values for this frame (updated inside face loop)
        ear            = ear_avg.get()
        mar            = mar_avg.get()
        perclos_val    = 0.0
        blinks_per_min = len(blinker._blink_times)
        alert_active   = False
        alert_reasons: list[str] = []

        for face in faces:
            shape = face_utils.shape_to_np(landmark_predictor(gray, face))

            left_eye = shape[le_s:le_e]
            right_eye = shape[re_s:re_e]
            mouth     = shape[m_s:m_e]

            raw_ear = (calculate_eye_aspect_ratio(left_eye) +
                       calculate_eye_aspect_ratio(right_eye)) / 2.0
            raw_mar = calculate_mouth_aspect_ratio(mouth)

            ear = ear_avg.update(raw_ear)
            mar = mar_avg.update(raw_mar)

            # Draw eye & mouth contours
            for pts in (left_eye, right_eye, mouth):
                cv2.drawContours(frame, [cv2.convexHull(pts)], -1, (0, 220, 0), 1)
            # Draw eye & mouth landmark dots only (keeps frame uncluttered)
            for idx in (list(range(le_s, le_e)) +
                        list(range(re_s, re_e)) +
                        list(range(m_s, m_e))):
                cv2.circle(frame, tuple(shape[idx]), 1, (0, 180, 0), -1)

            # ---- Calibration ----
            if calibrating:
                calib_ear_buf.append(raw_ear)
                calib_mar_buf.append(raw_mar)
                if len(calib_ear_buf) >= CALIBRATION_FRAMES:
                    ear_thr = np.mean(calib_ear_buf) * EAR_CALIB_FACTOR
                    mar_thr = np.mean(calib_mar_buf) * MAR_CALIB_FACTOR
                    ear_thr = float(np.clip(ear_thr, EAR_THRESHOLD_MIN, EAR_THRESHOLD_MAX))
                    mar_thr = float(np.clip(mar_thr, MAR_THRESHOLD_MIN, MAR_THRESHOLD_MAX))
                    calibrating = False
                    session_start = time.time()
                    print(f"Calibration done → EAR threshold={ear_thr:.3f}, "
                          f"MAR threshold={mar_thr:.3f}")
                    logger.log("CALIBRATION", ear_thr, mar_thr, 0.0, 0)
                break  # process only the first face during calibration

            # ---- PERCLOS ----
            perclos_val = perclos.update(ear, ear_thr)

            # ---- Blink rate ----
            blinks_per_min = blinker.update(ear, ear_thr)

            # ---- Head nod ----
            nod = nod_det.update(shape, face)

            # ---- Alert conditions ----
            # 1. Sustained eye closure (consecutive-frames method)
            if ear < ear_thr:
                eye_closed_frames += 1
                if eye_closed_frames >= EAR_CONSEC_FRAMES:
                    alert_reasons.append("EYES CLOSED")
            else:
                eye_closed_frames = 0

            # 2. PERCLOS threshold
            if perclos_val >= PERCLOS_ALERT_THRESHOLD:
                alert_reasons.append(f"PERCLOS {perclos_val*100:.0f}%")

            # 3. Yawn
            if mar > mar_thr:
                alert_reasons.append("YAWNING")

            # 4. Low blink rate (only after warm-up period)
            elapsed = now - session_start
            if elapsed >= BLINK_WARMUP_SEC and blinks_per_min < BLINK_MIN_PER_MIN:
                alert_reasons.append("LOW BLINK RATE")

            # 5. Head nodding
            if nod:
                alert_reasons.append("HEAD NOD")

            if alert_reasons:
                alert_active = True
                if alert_mgr.trigger():
                    logger.log(" | ".join(alert_reasons),
                               ear, mar, perclos_val, blinks_per_min)

            break  # process only the primary (first) face

        # ---- Draw HUD ----
        calib_progress = len(calib_ear_buf) / CALIBRATION_FRAMES
        draw_hud(frame, ear, mar, perclos_val, blinks_per_min, fps,
                 ear_thr, mar_thr, alert_active, alert_reasons,
                 calibrating, calib_progress)

        cv2.imshow("Drowsiness Detection System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting…")
            break
        elif key == ord("r"):
            calibrating = True
            calib_ear_buf.clear()
            calib_mar_buf.clear()
            eye_closed_frames = 0
            nod_det.reset_baseline()
            print("Recalibrating…")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Log saved to: {LOG_FILE}")
    print(f"Total blinks recorded: {blinker.total}")


if __name__ == "__main__":
    main()
