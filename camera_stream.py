"""
CameraProcessor — 카메라 캡처 + SCRFD 탐지 + ArcFace 인식 + INN 익명화
백그라운드 스레드에서 처리 후 MJPEG용 JPEG 버퍼를 유지한다.
"""
import io
import os
import threading
import sqlite3
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import config as c
from core.anonymizer import INNAnonymizer

DB_PATH = "security_system.db"
PENDING_DIR = "pending"  # 원본 프레임 임시 큐 (INN 대기)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu
    "C:/Windows/Fonts/NanumGothic.ttf",                  # Windows (나눔고딕 설치)
    "C:/Windows/Fonts/malgun.ttf",                        # Windows 맑은 고딕
    "C:/Windows/Fonts/gulim.ttc",                         # Windows 굴림
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _put_text(frame: np.ndarray, text: str, pos: tuple, size: int, color_bgr: tuple) -> np.ndarray:
    b, g, r = color_bgr
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    ImageDraw.Draw(pil).text(pos, text, font=_load_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# SQLite ↔ numpy 어댑터
def _adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return sqlite3.Binary(buf.read())


def _convert_array(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    buf.seek(0)
    return np.load(buf)


sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("array", _convert_array)


def _load_db() -> list:
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        rows = conn.execute("SELECT name, auth_group, vector FROM users").fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB] 로드 실패: {e}")
        return []


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


class CameraProcessor:
    """
    단일 카메라를 백그라운드 스레드로 처리.
    - SCRFD 탐지 → ArcFace 인식 → 사원/외부인 분기
    - 외부인: INN 익명화 + 빨간 박스
    - 사원: 권한 컬러 박스 + 이름
    - get_jpeg(): 최신 처리 프레임을 JPEG bytes로 반환
    """

    def __init__(self):
        print("[CameraProcessor] 모델 로드 중...")
        use_hailo = getattr(c, "USE_HAILO", False)
        if use_hailo:
            try:
                from hailo_infer import HAILO_AVAILABLE, HailoSCRFD, HailoArcFace
                if not HAILO_AVAILABLE:
                    raise RuntimeError("hailo_platform 미설치")
                self.detector = HailoSCRFD(
                    c.SCRFD_HEF_PATH,
                    conf_thresh=getattr(c, "HAILO_DET_THRESH", 0.5),
                )
                self.recognizer = HailoArcFace(c.ARCFACE_HEF_PATH)
                print("[CameraProcessor] ⚡ Hailo-8L 가속 사용 (SCRFD+ArcFace)")
            except Exception as e:
                print(f"[CameraProcessor] Hailo 사용 불가({e}) → insightface 폴백")
                use_hailo = False
        if not use_hailo:
            fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            fa.prepare(ctx_id=-1, det_thresh=0.6)
            self.detector = fa.models["detection"]
            self.recognizer = fa.models["recognition"]
            print("[CameraProcessor] insightface(CPU) 사용")

        if c.INN_CHECKPOINT:
            self._anonymizer = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
            print(f"[CameraProcessor] INN 로드: {c.INN_CHECKPOINT}")
        else:
            self._anonymizer = None
            print("[CameraProcessor] INN 체크포인트 없음 → 모자이크 익명화 사용")
        self._password = c.DEMO_PASSWORD

        self._db_lock = threading.Lock()
        self._db_users: list = []
        self.reload_db()

        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_raw_jpeg: bytes | None = None  # 익명화 전 원본 (등록용)

        self._tiles_lock = threading.Lock()
        self._latest_tiles: list = []   # [{"tile_f32": ndarray, "crop_box": list}]

        # 원본 프레임을 디스크 pending 큐에 저장 → recorder가 배치 INN 처리.
        # (모든 프레임을 놓치지 않고 저장; 실시간을 안 따라가도 결국 다 처리)
        self._pending_lock = threading.Lock()
        self._pending_seq = 0
        self._pending_count = 0
        import shutil
        shutil.rmtree(PENDING_DIR, ignore_errors=True)  # 이전 잔여 큐 정리
        os.makedirs(PENDING_DIR, exist_ok=True)
        # 대기 큐 상한 = 한 청크분(10분치). 넘으면 새 녹화를 일시 중단하고
        # recorder가 소진(청크 완성)할 때까지 기다림 → pending 폭발 방지.
        self._pending_max = (
            getattr(c, "CHUNK_MINUTES", 10) * 60 * getattr(c, "PROCESS_MAX_FPS", 15)
        )

        self._stats_lock = threading.Lock()
        self._stats = {"employee_count": 0, "unknown_count": 0, "recording": True}

        self._running = False
        print("[CameraProcessor] 준비 완료")

    # ── 공개 API ──────────────────────────────────────────────────────────

    def reload_db(self):
        with self._db_lock:
            self._db_users = _load_db()
        print(f"[DB] {len(self._db_users)}명 로드됨")

    def start(self, cam_id: int = 0):
        self._running = True
        t = threading.Thread(target=self._loop, args=(cam_id,), daemon=True)
        t.start()
        print(f"[CameraProcessor] 카메라 {cam_id} 시작")

    def stop(self):
        self._running = False

    def get_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def get_raw_jpeg(self) -> bytes | None:
        """익명화 전 원본 프레임 (사원 등록용)."""
        with self._frame_lock:
            return self._latest_raw_jpeg

    def capture_raw_frame(self) -> "np.ndarray | None":
        """현재 원본 프레임을 디코딩해 ndarray로 반환 (등록 처리용)."""
        jpeg = self.get_raw_jpeg()
        if jpeg is None:
            return None
        arr = np.frombuffer(jpeg, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def get_recording_snapshot(self) -> dict | None:
        """녹화용 스냅샷: 현재 JPEG + INN 타일 목록 반환."""
        with self._frame_lock:
            jpeg = self._latest_jpeg
        if jpeg is None:
            return None
        with self._tiles_lock:
            tiles = list(self._latest_tiles)
        return {"jpeg": jpeg, "tiles": tiles}

    # ── 캡처 루프 ─────────────────────────────────────────────────────────

    def get_debug_info(self) -> dict:
        with self._frame_lock:
            size = len(self._latest_jpeg) if self._latest_jpeg else 0
        return {
            "running": self._running,
            "jpeg_size": size,
            "has_frame": size > 0,
            "db_users": len(self._db_users),
            "stats": self.get_stats(),
        }

    # ── 전용 프레임 리더 (블로킹 cap.read 격리) ──────────────────────────────

    def _start_frame_reader(self, cap) -> "queue.Queue":
        """
        별도 스레드에서 cap.read()를 수행하고 결과를 Queue에 넣는다.
        Windows MSMF/DSHOW에서 cap.read()가 무한 블로킹하는 문제를 우회.
        """
        import queue
        q: "queue.Queue" = queue.Queue(maxsize=2)

        def _reader():
            while True:
                ret, frame = cap.read()
                try:
                    q.put_nowait((ret, frame))
                except Exception:
                    pass  # 큐가 가득 찬 경우 최신 프레임 우선 → 버림

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return q

    def _find_camera_index(self, preferred: int = 0) -> int:
        """
        컬러 영상(3채널, 밝기>3)이 나오는 카메라 인덱스를 자동 탐색.
        preferred를 먼저 시도하고, 안 되면 0~8을 스캔.
        RealSense는 depth/IR 등 여러 /dev/videoN 중 컬러 노드를 자동 선택.
        """
        candidates = [preferred] + [i for i in range(9) if i != preferred]
        for idx in candidates:
            try:
                cap = cv2.VideoCapture(idx)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            found = False
            for _ in range(15):
                ret, f = cap.read()
                if (ret and f is not None and f.ndim == 3
                        and f.shape[2] == 3 and float(f.mean()) > 3):
                    found = True
                    break
            cap.release()
            if found:
                print(f"[Camera] 자동 선택: 인덱스 {idx} (컬러 영상 확인)")
                return idx
        print(f"[Camera] 컬러 카메라 자동탐색 실패 → 인덱스 {preferred} 사용")
        return preferred

    def _loop(self, cam_id: int):
        import time

        # FORCE_VIDEO: 카메라 대신 폴백 영상 사용
        if getattr(c, "FORCE_VIDEO", False):
            fallback = getattr(c, "VIDEO_FALLBACK", None)
            if fallback and os.path.exists(fallback):
                print(f"[Camera] FORCE_VIDEO=True → 영상 재생: {fallback}")
                self._video_loop(fallback)
                return

        # RealSense 카메라면 전용 루프
        if getattr(c, "CAMERA_TYPE", "webcam") == "realsense":
            self._realsense_loop()
            return

        # ── 컬러 영상이 나오는 카메라 인덱스 자동 선택 ──
        cam_id = self._find_camera_index(cam_id)

        # ── 카메라 열기 ──
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[Camera] 카메라 {cam_id} 열기 실패 → 5초 후 재시도")
            self._show_reconnecting()
            time.sleep(5)
            self._loop(cam_id)
            return
        # 버퍼 최소화 (지연 감소)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        print(f"[Camera] 카메라 {cam_id} 열림")

        # ── 리더 스레드: 항상 '최신' 프레임만 보관 (버퍼 누적 지연 제거) ──
        state = {"frame": None, "run": True, "first": True}
        rlock = threading.Lock()

        # pending 저장 fps 상한 (PROCESS_MAX_FPS). 0이면 카메라 fps 전부.
        save_fps = getattr(c, "PROCESS_MAX_FPS", 15)
        save_dt = (1.0 / save_fps) if save_fps and save_fps > 0 else 0.0

        def _reader():
            last_save = 0.0
            while state["run"] and self._running:
                ret, f = cap.read()
                if not ret or f is None:
                    continue
                f = cv2.flip(f, 1)
                # 등록용 원본은 매 프레임 갱신
                _okr, _bufr = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if _okr:
                    with self._frame_lock:
                        self._latest_raw_jpeg = _bufr.tobytes()
                with rlock:
                    state["frame"] = f  # 실시간 화면용 최신 프레임

                # pending 큐에 원본 저장 (fps 상한). 단, 큐가 한 청크분 이상
                # 쌓이면 저장 중단 → recorder가 소진(청크 완성)할 때까지 대기.
                now = time.time()
                if (save_dt == 0.0 or (now - last_save) >= save_dt) \
                        and self._pending_count < self._pending_max:
                    last_save = now
                    self._save_pending(f, now)

        threading.Thread(target=_reader, daemon=True).start()

        # 처리 fps 상한(시간 기반) — 카메라 버퍼 폭주로 초당 수십 장을
        # 저장/처리하는 것을 방지. 화면·저장·큐를 이 fps로 안정화.
        max_fps = getattr(c, "PROCESS_MAX_FPS", 15)
        min_dt = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0
        last_proc = 0.0
        while self._running:
            with rlock:
                frame = state["frame"]
                state["frame"] = None  # 소비 → 다음 새 프레임까지 대기
            if frame is None:
                time.sleep(0.005)
                continue

            if state["first"]:
                state["first"] = False
                print(f"[Camera] 첫 프레임 수신 {frame.shape}")

            # fps 상한: 너무 빠르면 이 프레임은 버리고 최신만 처리
            now = time.time()
            if min_dt > 0 and (now - last_proc) < min_dt:
                time.sleep(0.002)
                continue
            last_proc = now

            frame = self._maybe_downscale(frame)
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Camera] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

        state["run"] = False
        cap.release()
        print(f"[Camera] 카메라 {cam_id} 종료")

    def _realsense_loop(self):
        """
        Intel RealSense (D455 등) 컬러 스트림으로 프레임을 받아 처리.
        cv2.VideoCapture 대신 pyrealsense2 파이프라인 사용.
        """
        import time
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("[Camera] pyrealsense2 미설치 → 'pip install pyrealsense2'")
            self._show_reconnecting()
            return

        W = getattr(c, "REALSENSE_WIDTH", 640)
        H = getattr(c, "REALSENSE_HEIGHT", 480)
        FPS = getattr(c, "REALSENSE_FPS", 30)

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

        try:
            pipeline.start(cfg)
        except Exception as e:
            print(f"[Camera] RealSense 시작 실패: {e} → 5초 후 재시도")
            self._show_reconnecting()
            time.sleep(5)
            self._realsense_loop()
            return

        print(f"[Camera] RealSense 시작 ({W}x{H} @ {FPS}fps)")
        every_n = max(1, getattr(c, "PROCESS_EVERY_N", 1))
        frame_count = 0
        try:
            while self._running:
                try:
                    frames = pipeline.wait_for_frames(2000)  # 2초 타임아웃
                except Exception:
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue

                frame = np.asanyarray(color.get_data())  # HWC BGR uint8
                frame_count += 1
                if frame_count == 1:
                    print(f"[Camera] RealSense 첫 프레임 {frame.shape}")

                # 익명화 전 원본 저장 (등록용) — 매 프레임
                _okr, _bufr = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if _okr:
                    with self._frame_lock:
                        self._latest_raw_jpeg = _bufr.tobytes()

                # N프레임마다만 무거운 처리
                if frame_count % every_n != 0:
                    continue

                frame = self._maybe_downscale(frame)
                try:
                    frame, emp, unk = self._process(frame)
                except Exception as e:
                    print(f"[Camera] _process 오류 (건너뜀): {e}")
                    emp, unk = 0, 0

                with self._stats_lock:
                    self._stats["employee_count"] = emp
                    self._stats["unknown_count"] = unk

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = buf.tobytes()
        finally:
            pipeline.stop()
            print("[Camera] RealSense 종료")

    def _video_loop(self, video_path: str):
        """
        폴백: mp4 등 동영상 파일을 카메라처럼 처리.
        - 각 프레임에 탐지/인식/익명화 적용 (카메라와 동일)
        - 영상이 끝나면 처음부터 다시 재생 (무한 루프)
        """
        import time

        fps = getattr(c, "VIDEO_FALLBACK_FPS", 25)
        delay = 1.0 / max(1, fps)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Video] 영상 열기 실패: {video_path} → 더미 모드")
            self._dummy_loop()
            return

        print(f"[Video] 폴백 영상 재생 시작 (목표 fps={fps})")
        frame_count = 0
        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                # 영상 끝 → 처음으로 되감기
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Video] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0

            # 폴백 영상 표시 (데모용 표식)
            cv2.putText(frame, f"DEMO (video) F{frame_count}",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

            # 처리 시간만큼 빼서 보정 — 처리가 이미 느리면 sleep 안 함
            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)

        cap.release()
        print("[Video] 폴백 영상 종료")

    def _show_reconnecting(self):
        """재연결 대기 중 화면을 JPEG 버퍼에 공급."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)
        cv2.putText(frame, "Reconnecting...", (160, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 140, 200), 2)
        cv2.putText(frame, "Camera disconnected. Retrying in 5s", (60, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", frame)
        with self._frame_lock:
            self._latest_jpeg = buf.tobytes()

    def _dummy_loop(self):
        """카메라 없을 때 대기 화면을 MJPEG 버퍼에 계속 공급."""
        import time
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)  # 어두운 배경
        cv2.putText(frame, "No Camera", (200, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 120), 2)
        cv2.putText(frame, "Connect webcam & restart server", (70, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", frame)
        jpeg = buf.tobytes()
        with self._frame_lock:
            self._latest_jpeg = jpeg
        # 더미 프레임은 갱신할 필요 없으므로 대기만 함
        while self._running:
            time.sleep(1)

    # ── 모자이크 익명화 (INN 대체 fallback) ──────────────────────────────

    def _mosaic(self, frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        out = frame.copy()
        roi = out[y1:y2, x1:x2]
        if roi.size > 0:
            h, w = roi.shape[:2]
            small = cv2.resize(roi, (max(1, w // 8), max(1, h // 8)))
            out[y1:y2, x1:x2] = cv2.resize(
                small, (w, h), interpolation=cv2.INTER_NEAREST
            )
        return out

    # ── 프레임 처리 ───────────────────────────────────────────────────────

    def _maybe_downscale(self, frame: np.ndarray) -> np.ndarray:
        """PROCESS_WIDTH 설정 시 처리 속도를 위해 프레임 축소."""
        pw = getattr(c, "PROCESS_WIDTH", 0)
        if pw and frame.shape[1] > pw:
            scale = pw / frame.shape[1]
            frame = cv2.resize(frame, (pw, int(frame.shape[0] * scale)))
        return frame

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """실시간 디스플레이 전용. 화면은 가벼운 모자이크로 익명화 (INN 안 씀)."""
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return frame, 0, 0

        anonymize_all = getattr(c, "ANONYMIZE_ALL", False)
        emp, unk = 0, 0
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            lm = kpss[i]
            aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
            emb = self.recognizer.get_feat(aligned)
            name, group, sim = self._match(emb)

            if name == "Unknown":
                unk += 1
            else:
                emp += 1

            if name == "Unknown" or anonymize_all:
                frame = self._mosaic(frame, x1, y1, x2, y2)
            frame = self._draw(frame, x1, y1, x2, y2, name, group, sim)
        return frame, emp, unk

    # ── pending 디스크 큐 (모든 원본 프레임 → recorder가 배치 INN) ──────────

    def _save_pending(self, frame: np.ndarray, ts: float):
        """원본 프레임 + 시각을 pending 폴더에 저장 (INN 대기 큐)."""
        with self._pending_lock:
            self._pending_seq += 1
            self._pending_count += 1
            seq = self._pending_seq
        d = os.path.join(PENDING_DIR, f"{seq:09d}")
        try:
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "frame.jpg"), frame)
            with open(os.path.join(d, "ts.txt"), "w") as f:
                f.write(repr(ts))
            open(os.path.join(d, "ready"), "w").close()  # 쓰기 완료 표시
        except Exception as e:
            print(f"[Pending] 저장 실패: {e}")

    def pop_pending(self):
        """가장 오래된 완성 pending 항목의 (frame, ts) 반환 후 삭제. 없으면 None."""
        import shutil
        try:
            items = sorted(
                d for d in os.listdir(PENDING_DIR)
                if os.path.exists(os.path.join(PENDING_DIR, d, "ready"))
            )
        except FileNotFoundError:
            return None
        if not items:
            return None
        d = os.path.join(PENDING_DIR, items[0])
        frame = cv2.imread(os.path.join(d, "frame.jpg"))
        ts = 0.0
        try:
            with open(os.path.join(d, "ts.txt")) as f:
                ts = float(f.read())
        except Exception:
            pass
        shutil.rmtree(d, ignore_errors=True)
        with self._pending_lock:
            self._pending_count = max(0, self._pending_count - 1)
        if frame is None:
            return None
        return frame, ts

    def pending_size(self) -> int:
        try:
            return sum(
                1 for d in os.listdir(PENDING_DIR)
                if os.path.exists(os.path.join(PENDING_DIR, d, "ready"))
            )
        except FileNotFoundError:
            return 0

    def oldest_pending_ts(self) -> float | None:
        """아직 처리 안 된 가장 오래된 대기 프레임의 촬영 시각. 큐 비면 None."""
        try:
            items = sorted(
                d for d in os.listdir(PENDING_DIR)
                if os.path.exists(os.path.join(PENDING_DIR, d, "ready"))
            )
        except FileNotFoundError:
            return None
        if not items:
            return None
        try:
            with open(os.path.join(PENDING_DIR, items[0], "ts.txt")) as f:
                return float(f.read())
        except Exception:
            return None

    def make_protected(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        원본 프레임을 detect + INN protect 해 보호본 생성 (recorder 배치용).
        Returns: (보호본 프레임, tiles)  얼굴 없으면 (원본, [])
        """
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return frame, []
        anonymize_all = getattr(c, "ANONYMIZE_ALL", False)
        out = frame
        tiles = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            lm = kpss[i]
            aligned = face_align.norm_crop(out, landmark=lm, image_size=112)
            emb = self.recognizer.get_feat(aligned)
            name, group, sim = self._match(emb)
            if name == "Unknown" or anonymize_all:
                if self._anonymizer is not None:
                    try:
                        out, tile_f32, crop_box = self._anonymizer.protect_roi(
                            out, [x1, y1, x2, y2], self._password
                        )
                        tiles.append({"tile_f32": tile_f32, "crop_box": crop_box})
                    except Exception as e:
                        print(f"[INN] make_protected 실패 → 모자이크: {e}")
                        out = self._mosaic(out, x1, y1, x2, y2)
                else:
                    out = self._mosaic(out, x1, y1, x2, y2)
            out = self._draw(out, x1, y1, x2, y2, name, group, sim)
        return out, tiles

    def _match(self, emb) -> tuple[str, str, float]:
        best_name, best_group, best_sim = "Unknown", "비허가", -1.0
        with self._db_lock:
            if emb is not None and self._db_users:
                for db_name, db_group, db_vec in self._db_users:
                    s = _cosine_sim(emb, db_vec)
                    if s > best_sim:
                        best_sim = s
                        if s > c.MATCH_THRESHOLD:
                            best_name = db_name
                            best_group = db_group
        return best_name, best_group, best_sim

    def _draw(self, frame: np.ndarray, x1, y1, x2, y2, name, group, sim) -> np.ndarray:
        if group == "허가":
            color = (0, 200, 0)
        elif group == "준허가":
            color = (0, 200, 200)
        else:
            color = (0, 0, 220)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"외부인 ({sim:.2f})" if name == "Unknown" else f"{name} ({sim:.2f})"
        text_y = max(y1 - 28, 5)
        frame = _put_text(frame, label, (x1, text_y), 18, color)
        return frame
