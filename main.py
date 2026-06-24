"""
SecureFace-RX v2 — 통합 FastAPI 서버

  GET  /                → 실시간 모니터링 (SCR-002)
  GET  /register        → 사원 등록 UI
  GET  /stream/cam_0    → MJPEG 익명화 스트림
  GET  /api/stats       → 실시간 통계 (사원수/외부인수)
  GET  /api/users       → 등록 사원 목록
  POST /api/users/reload → 카메라 DB 즉시 갱신
  POST /api/register    → 얼굴 등록 (3각도)

실행:
  python main.py
  또는
  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import base64
import os
import sqlite3
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from insightface.utils import face_align
from pydantic import BaseModel

import config as c
from camera_stream import (
    CameraProcessor,
    DB_PATH,
    _adapt_array,
    _convert_array,
)

IMAGE_DIR = "registered_faces"
os.makedirs(IMAGE_DIR, exist_ok=True)

camera: CameraProcessor | None = None


# ── DB 초기화 ─────────────────────────────────────────────────────────────
def _init_db():
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            auth_group TEXT NOT NULL,
            image_path TEXT NOT NULL,
            vector     array NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ── Lifespan (FastAPI 0.93+ 권장 방식) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera
    _init_db()
    camera = CameraProcessor()
    camera.start(cam_id=0)
    yield
    if camera:
        camera.stop()


# ── 앱 초기화 ─────────────────────────────────────────────────────────────
app = FastAPI(title="SecureFace-RX v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")


# ── 페이지 ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def page_monitor(request: Request):
    return templates.TemplateResponse(request=request, name="monitor.html")


@app.get("/register", response_class=HTMLResponse)
async def page_register(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ── 스냅샷 (단일 JPEG, JS 폴링용) ────────────────────────────────────────
@app.get("/snapshot/cam_0")
async def snapshot_cam0():
    jpeg = camera.get_jpeg() if camera else None
    if jpeg is None:
        # 카메라 준비 중 — 빈 1×1 회색 JPEG 반환
        import base64
        placeholder = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
            "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
            "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
            "/9oADAMBAAIRAxEAPwCwABmX/9k="
        )
        return Response(content=placeholder, media_type="image/jpeg")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )

# ── MJPEG 스트리밍 (호환 브라우저용, 유지) ───────────────────────────────
@app.get("/stream/cam_0")
async def stream_cam0():
    async def generate():
        while True:
            jpeg = camera.get_jpeg() if camera else None
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── API — 통계 ────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def api_stats():
    return camera.get_stats() if camera else {"employee_count": 0, "unknown_count": 0}


# ── API — 디버그 (카메라 상태 확인) ──────────────────────────────────────
@app.get("/api/debug")
async def api_debug():
    if camera is None:
        return {"error": "camera not initialized"}
    return camera.get_debug_info()


# ── API — 사원 목록 ───────────────────────────────────────────────────────
@app.get("/api/users")
async def api_users():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT SUBSTR(name, 1, INSTR(name||'_', '_') - 1) as base_name, "
        "auth_group FROM users"
    ).fetchall()
    conn.close()
    seen = {}
    for base_name, group in rows:
        seen[base_name] = group
    return [{"name": k, "group": v} for k, v in seen.items()]


# ── API — DB 재로드 ───────────────────────────────────────────────────────
@app.post("/api/users/reload")
async def api_reload():
    if camera:
        camera.reload_db()
    return {"status": "ok"}


# ── API — 얼굴 등록 ───────────────────────────────────────────────────────
class RegisterData(BaseModel):
    name: str
    group: str
    image_base64: str


@app.post("/api/register")
async def api_register(data: RegisterData):
    try:
        if camera is None:
            return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}

        detector = camera.detector
        recognizer = camera.recognizer

        _, encoded = data.image_base64.split(",", 1)
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR,
        )

        bboxes, kpss = detector.detect(frame, max_num=1, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return {"status": "error", "message": "❌ 얼굴을 찾을 수 없습니다."}

        x1, y1, x2, y2 = bboxes[0, :4].astype(int).tolist()
        lm = kpss[0]

        # 측면 판별 (눈-코 거리 비율)
        le, re, nose = lm[0], lm[1], lm[2]
        d_left = np.linalg.norm(le - nose)
        d_right = np.linalg.norm(re - nose)
        ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
        eye_dist = np.linalg.norm(le - re)
        is_side = ratio > 1.5 or (eye_dist / (x2 - x1 + 1e-5)) < 0.25

        aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
        emb = recognizer.get_feat(aligned)
        if emb is None:
            return {"status": "error", "message": "❌ 임베딩 추출 실패"}

        if not is_side:
            tag, fname = "정면", f"{data.name}_정면.jpg"
        else:
            if d_left < d_right:
                tag, fname = "좌측면", f"{data.name}_측면1(좌).jpg"
            else:
                tag, fname = "우측면", f"{data.name}_측면2(우).jpg"

        fpath = os.path.join(IMAGE_DIR, fname)
        cv2.imwrite(fpath, frame)

        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute(
            "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?,?,?,?)",
            (f"{data.name}_{tag}", data.group, fpath, emb),
        )
        conn.commit()
        conn.close()

        camera.reload_db()
        return {"status": "success", "message": f"✅ [{data.name}] {tag} 등록 성공!"}

    except Exception as e:
        return {"status": "error", "message": f"서버 오류: {e}"}


# ── 직접 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
