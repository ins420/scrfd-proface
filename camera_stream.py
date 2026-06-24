"""
CameraProcessor — 카메라 캡처 + SCRFD 탐지 + ArcFace 인식 + INN 익명화
백그라운드 스레드에서 처리 후 MJPEG용 JPEG 버퍼를 유지한다.
"""
import io
import os
import threading
import sqlite3

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import config as c
from core.anonymizer import INNAnonymizer

DB_PATH = "security_system.db"

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
        fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_thresh=0.6)
        self.detector = fa.models["detection"]
        self.recognizer = fa.models["recognition"]

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

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

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

    @staticmethod
    def _probe(cap, n: int = 20, timeout: float = 5.0) -> bool:
        """
        별도 스레드에서 최대 n 프레임 읽기. 평균 밝기>5인 프레임이
        하나라도 나오면 True. timeout 초 내 응답 없으면 False.
        MSMF 무한 블로킹 방지용.
        """
        import queue as _q, threading as _t
        result: "_q.Queue[bool]" = _q.Queue(1)

        def _r():
            found = False
            for _ in range(n):
                try:
                    ret, f = cap.read()
                    if ret and f is not None and float(f.mean()) > 5:
                        found = True
                        break
                except Exception:
                    break
            result.put(found)

        _t.Thread(target=_r, daemon=True).start()
        try:
            return result.get(timeout=timeout)
        except _q.Empty:
            return False

    def _loop(self, cam_id: int):
        import time
        import queue as _queue

        def _setup(cap_obj):
            cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def _can_read(cap_obj, timeout=4.0) -> bool:
            """프레임을 하나라도 읽으면 True (밝기 무관)."""
            import queue as _q, threading as _t
            result: "_q.Queue[bool]" = _q.Queue(1)
            def _r():
                for _ in range(10):
                    ret, f = cap_obj.read()
                    if ret and f is not None:
                        result.put(True)
                        return
                result.put(False)
            _t.Thread(target=_r, daemon=True).start()
            try:
                return result.get(timeout=timeout)
            except _q.Empty:
                return False

        cap = None
        found_id = -1

        # ① 장치 이름으로 DSHOW 직접 열기
        dev_name = getattr(c, "CAMERA_DEVICE_NAME", None)
        if dev_name:
            dshow_str = f"video={dev_name}"
            print(f"[Camera] 이름으로 열기: '{dshow_str}'")
            c_try = cv2.VideoCapture(dshow_str, cv2.CAP_DSHOW)
            if c_try.isOpened() and _can_read(c_try):
                _setup(c_try)
                cap = c_try
                found_id = dshow_str
                print(f"[Camera] '{dshow_str}' 오픈 성공!")
            else:
                print(f"[Camera] '{dshow_str}' 실패 → 인덱스 탐색으로 fallback")
                c_try.release()

        # ② 인덱스 탐색 (DSHOW → MSMF, 밝기 무관하게 읽히면 OK)
        if cap is None:
            BACKENDS = [(cv2.CAP_DSHOW, "DSHOW"), (None, "MSMF")]
            ids_to_try = [cam_id] + [i for i in range(5) if i != cam_id]
            for idx in ids_to_try:
                for backend, bname in BACKENDS:
                    try:
                        c_try = (
                            cv2.VideoCapture(idx, backend)
                            if backend is not None
                            else cv2.VideoCapture(idx)
                        )
                        if not c_try.isOpened():
                            c_try.release()
                            continue
                        _setup(c_try)
                        print(f"[Camera] idx={idx} ({bname}) 읽기 테스트...")
                        if _can_read(c_try):
                            cap = c_try
                            found_id = idx
                            print(f"[Camera] 카메라 {idx} ({bname}) 선택됨!")
                            break
                        else:
                            print(f"[Camera] idx={idx} ({bname}) 읽기 불가 → 건너뜀")
                            c_try.release()
                    except Exception as e:
                        print(f"[Camera] idx={idx} ({bname}) 오류: {e}")
                if cap is not None:
                    break

        if cap is None:
            print("[Camera] 사용 가능한 카메라 없음 → 더미 프레임 모드")
            self._dummy_loop()
            return

        print(f"[Camera] 카메라 확정: {found_id}")

        # 전용 리더 스레드 시작 (cap.read 블로킹 격리)
        frame_q = self._start_frame_reader(cap)

        fail_count = 0
        frame_count = 0
        while self._running:
            try:
                ret, frame = frame_q.get(timeout=2.0)
            except _queue.Empty:
                fail_count += 1
                print(f"[Camera] 프레임 타임아웃 ({fail_count}회)")
                if fail_count > 10:
                    print("[Camera] 연속 타임아웃 → 더미 프레임 모드")
                    cap.release()
                    self._dummy_loop()
                    return
                continue

            if not ret or frame is None:
                fail_count += 1
                print(f"[Camera] 프레임 읽기 실패 ({fail_count}회)")
                if fail_count > 30:
                    print("[Camera] 연속 실패 → 더미 프레임 모드")
                    cap.release()
                    self._dummy_loop()
                    return
                time.sleep(0.05)
                continue

            fail_count = 0
            frame_count += 1
            if frame_count == 1:
                print(f"[Camera] 첫 프레임 수신 {frame.shape}")
                cv2.imwrite("debug_raw_frame.jpg", frame)
                print("[Camera] debug_raw_frame.jpg 저장됨 (카메라 진단용)")

            frame = cv2.flip(frame, 1)
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Camera] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0

            # 화면이 검은지 확인용 — 항상 우상단에 프레임 번호 표시
            cv2.putText(frame, f"F{frame_count}", (frame.shape[1]-60, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

        cap.release()
        print(f"[Camera] 카메라 {cam_id} 종료")

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

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return frame, 0, 0

        emp, unk = 0, 0
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            lm = kpss[i]

            aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
            emb = self.recognizer.get_feat(aligned)
            name, group, sim = self._match(emb)

            if name == "Unknown":
                unk += 1
                if self._anonymizer is not None:
                    try:
                        frame, _, _ = self._anonymizer.protect_roi(
                            frame, [x1, y1, x2, y2], self._password
                        )
                    except Exception as e:
                        print(f"[INN] protect_roi 실패 → 모자이크: {e}")
                        frame = self._mosaic(frame, x1, y1, x2, y2)
                else:
                    frame = self._mosaic(frame, x1, y1, x2, y2)
            else:
                emp += 1

            frame = self._draw(frame, x1, y1, x2, y2, name, group, sim)

        return frame, emp, unk

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
