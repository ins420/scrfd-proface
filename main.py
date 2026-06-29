"""
SecureFace-RX v2 — 통합 FastAPI 서버

  GET  /                        → 실시간 모니터링 (SCR-002)
  GET  /register                → 사원 등록 UI
  GET  /assets                  → 보호 자산 목록 (SCR-003/004)
  GET  /employees               → 사원 관리 UI (SCR-006)
  GET  /stream/cam_0            → MJPEG 익명화 스트림
  GET  /api/stats               → 실시간 통계
  GET  /api/users               → 등록 사원 목록
  DELETE /api/users/{name}      → 사원 삭제 (SCR-006)
  POST /api/users/reload        → 카메라 DB 즉시 갱신
  POST /api/register            → 얼굴 등록 (3각도)
  GET  /api/assets              → 녹화 청크 목록
  GET  /api/assets/{chunk_id}   → 청크 상세
  GET  /recordings/{chunk_id}/thumb            → 청크 썸네일 JPEG
  GET  /recordings/{chunk_id}/{frame_id}/frame → 개별 프레임 JPEG
  POST /api/restore             → INN 역변환 복원 (SCR-005)

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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
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
from recorder import PSFRecorder

IMAGE_DIR = "registered_faces"
os.makedirs(IMAGE_DIR, exist_ok=True)

camera: CameraProcessor | None = None
recorder: PSFRecorder | None = None


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
    global camera, recorder
    _init_db()
    camera = CameraProcessor()
    camera.start(cam_id=getattr(c, "CAMERA_INDEX", 0))
    recorder = PSFRecorder(camera, interval_sec=getattr(c, "RECORD_INTERVAL", 5))
    recorder.start()
    yield
    if recorder:
        recorder.stop()
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


@app.get("/assets", response_class=HTMLResponse)
async def page_assets(request: Request):
    return templates.TemplateResponse(request=request, name="assets.html")


@app.get("/employees", response_class=HTMLResponse)
async def page_employees(request: Request):
    return templates.TemplateResponse(request=request, name="employees.html")


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


# ── API — 사원 삭제 (Phase 6) ─────────────────────────────────────────────
@app.delete("/api/users/{name}")
async def api_delete_user(name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "DELETE FROM users WHERE name LIKE ?", (f"{name}_%",)
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="사원을 찾을 수 없습니다.")
    if camera:
        camera.reload_db()
    return {"status": "ok", "deleted": deleted}


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


# ── API — 보호 자산 목록 (Phase 4) ───────────────────────────────────────
@app.get("/api/assets")
async def api_assets():
    if recorder is None:
        return []
    return recorder.list_chunks()


@app.get("/api/assets/{chunk_id}")
async def api_asset_detail(chunk_id: str):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    detail = recorder.get_chunk_detail(chunk_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="청크를 찾을 수 없습니다.")
    return detail


@app.get("/recordings/{chunk_id}/thumb")
async def recording_thumb(chunk_id: str):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    data = recorder.get_thumb_jpeg(chunk_id)
    if data is None:
        raise HTTPException(status_code=404, detail="썸네일 없음")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=60"})


@app.get("/recordings/{chunk_id}/{frame_id}/frame")
async def recording_frame(chunk_id: str, frame_id: str):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    data = recorder.get_frame_jpeg(chunk_id, frame_id)
    if data is None:
        raise HTTPException(status_code=404, detail="프레임 없음")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


# ── API — INN 복원 (Phase 5) ──────────────────────────────────────────────
class RestoreRequest(BaseModel):
    chunk_id: str
    frame_id: str
    password: str


@app.post("/api/restore")
async def api_restore(req: RestoreRequest):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    data = recorder.restore_frame(req.chunk_id, req.frame_id, req.password)
    if data is None:
        raise HTTPException(status_code=404, detail="프레임을 찾을 수 없습니다.")
    encoded = base64.b64encode(data).decode()
    return {"status": "ok", "image_base64": f"data:image/jpeg;base64,{encoded}"}


# ── API — 청크 전체 복원 영상 ─────────────────────────────────────────────
class RestoreVideoRequest(BaseModel):
    chunk_id: str
    password: str


@app.post("/api/restore_video")
async def api_restore_video(req: RestoreVideoRequest):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    path = await asyncio.to_thread(
        recorder.restore_chunk_video, req.chunk_id, req.password
    )
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="복원할 프레임이 없습니다.")
    return FileResponse(path, media_type="video/mp4", filename="restored.mp4")


# ── 직접 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
