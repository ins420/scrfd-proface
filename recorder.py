"""
PSFRecorder — 1시간 단위 PSF 청크 녹화 (Phase 3)

청크 구조:
  recordings/
    YYYY-MM-DD_HH/        ← 1시간 단위 청크
      manifest.json
      000001/             ← 스냅샷 (N초 간격)
        frame.jpg         ← 익명화된 프레임
        face_0.npy        ← float32 (3,256,256) 복원 타일
        face_0_box.json   ← crop_box [x1,y1,x2,y2]
      000002/
        ...
"""

import hashlib
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

RECORDINGS_DIR = "recordings"


# ── JSON 유틸 ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_frame_jpg(chunk_path: str) -> str | None:
    for name in sorted(os.listdir(chunk_path)):
        fdir = os.path.join(chunk_path, name)
        fpath = os.path.join(fdir, "frame.jpg")
        if os.path.isdir(fdir) and os.path.exists(fpath):
            return fpath
    return None


# ── PSFRecorder ────────────────────────────────────────────────────────────

class PSFRecorder:
    """
    CameraProcessor에서 N초 간격으로 스냅샷을 받아 PSF 청크로 저장.
    1시간마다 새 청크 폴더 자동 생성.
    """

    def __init__(self, camera, interval_sec: int = 5):
        self._camera = camera
        self._interval = interval_sec
        self._running = False
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"[Recorder] 녹화 시작 (간격={self._interval}s)")

    def stop(self):
        self._running = False

    # ── 내부 루프 ──────────────────────────────────────────────────────────

    def _chunk_dir(self) -> str:
        path = os.path.join(RECORDINGS_DIR, datetime.now().strftime("%Y-%m-%d_%H"))
        os.makedirs(path, exist_ok=True)
        return path

    def _loop(self):
        frame_id = 0
        while self._running:
            time.sleep(self._interval)

            snap = self._camera.get_recording_snapshot()
            if snap is None:
                continue

            chunk = self._chunk_dir()
            frame_id += 1
            snap_dir = os.path.join(chunk, f"{frame_id:06d}")
            os.makedirs(snap_dir, exist_ok=True)

            # frame.jpg 저장
            arr = np.frombuffer(snap["jpeg"], np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frame_path = os.path.join(snap_dir, "frame.jpg")
            cv2.imwrite(frame_path, img)

            # 타일 저장
            for i, td in enumerate(snap.get("tiles", [])):
                npy_path = os.path.join(snap_dir, f"face_{i}.npy")
                box_path = os.path.join(snap_dir, f"face_{i}_box.json")
                np.save(npy_path, td["tile_f32"])
                _save_json(box_path, td["crop_box"])

            # manifest 업데이트
            mpath = os.path.join(chunk, "manifest.json")
            m = _load_json(mpath) or {
                "chunk_id": os.path.basename(chunk),
                "start_time": datetime.now().isoformat(),
                "frame_count": 0,
                "total_faces": 0,
            }
            m["frame_count"] += 1
            m["total_faces"] += len(snap.get("tiles", []))
            m["last_update"] = datetime.now().isoformat()
            _save_json(mpath, m)

    # ── 공개 API ──────────────────────────────────────────────────────────

    def list_chunks(self) -> list[dict]:
        """녹화된 청크 목록 (최신순)."""
        result = []
        if not os.path.exists(RECORDINGS_DIR):
            return result
        for name in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            path = os.path.join(RECORDINGS_DIR, name)
            mpath = os.path.join(path, "manifest.json")
            if not (os.path.isdir(path) and os.path.exists(mpath)):
                continue
            m = _load_json(mpath) or {}
            m["chunk_id"] = name
            m["has_thumb"] = _first_frame_jpg(path) is not None
            result.append(m)
        return result

    def get_chunk_detail(self, chunk_id: str) -> dict | None:
        path = os.path.join(RECORDINGS_DIR, chunk_id)
        mpath = os.path.join(path, "manifest.json")
        if not os.path.exists(mpath):
            return None
        m = _load_json(mpath) or {}
        m["chunk_id"] = chunk_id
        frames = []
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fdir = os.path.join(path, fname)
                if not (os.path.isdir(fdir) and fname.isdigit()):
                    continue
                files = os.listdir(fdir)
                npys = [f for f in files if f.endswith(".npy")]
                frames.append({
                    "frame_id": fname,
                    "face_count": len(npys),
                    "has_faces": len(npys) > 0,
                })
        m["frames"] = frames
        return m

    def get_thumb_jpeg(self, chunk_id: str) -> bytes | None:
        path = os.path.join(RECORDINGS_DIR, chunk_id)
        jpg = _first_frame_jpg(path)
        if jpg is None:
            return None
        with open(jpg, "rb") as f:
            return f.read()

    def get_frame_jpeg(self, chunk_id: str, frame_id: str) -> bytes | None:
        p = os.path.join(RECORDINGS_DIR, chunk_id, frame_id, "frame.jpg")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def restore_frame(
        self, chunk_id: str, frame_id: str, password: str
    ) -> bytes | None:
        """INN 역변환으로 익명화 얼굴 복원 → JPEG bytes 반환."""
        import config as c
        from core.anonymizer import INNAnonymizer

        snap_dir = os.path.join(RECORDINGS_DIR, chunk_id, frame_id)
        frame_path = os.path.join(snap_dir, "frame.jpg")
        if not os.path.exists(frame_path):
            return None

        frame = cv2.imread(frame_path)
        if frame is None:
            return None

        if c.INN_CHECKPOINT is None:
            # 체크포인트 없음 → 원본 익명화 프레임 그대로 반환 + 워터마크
            cv2.putText(frame, "INN checkpoint required for restoration",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes() if ok else None

        anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
        for i in range(20):
            npy_path = os.path.join(snap_dir, f"face_{i}.npy")
            box_path = os.path.join(snap_dir, f"face_{i}_box.json")
            if not os.path.exists(npy_path):
                break
            tile_f32 = np.load(npy_path)
            crop_box = _load_json(box_path)
            try:
                frame = anon.restore_roi(frame, tile_f32, crop_box, password)
            except Exception as e:
                print(f"[Restore] face_{i} 복원 실패: {e}")

        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None
