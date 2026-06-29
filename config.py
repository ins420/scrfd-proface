"""
SecureFace-RX 전역 설정
실제 ProFace S config/config.py 기준으로 작성
"""

import os
import torch

# ─── 디바이스 ─────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ─── INN 아키텍처 ─────────────────────────────────────────────────
INV_BLOCKS   = 3       # INV_block_affine 반복 수 (config.INV_BLOCKS)
channels_in  = 3       # 입력 채널 수 (RGB)
clamp        = 2.0     # affine 스케일 클램핑 계수

# ─── 오복원(Wrong Recovery) 모드 ──────────────────────────────────
# 'Random': RandWR — 랜덤 노이즈형 오복원 (PSNR<11dB)
# 'Obfs'  : ObfsWR — 난독화 유지형 오복원
WRONG_RECOVER_TYPE = 'Random'

# ─── 키 보조입력 정책 ─────────────────────────────────────────────
SECRET_KEY_AS_NOISE = True  # 복원 보조입력으로 K를 3채널 반복

# ─── Utility 조건부 기능 (기본 비활성) ───────────────────────────
ADJ_UTILITY = False

# ─── 정규화 해상도 ────────────────────────────────────────────────
# 원본 config: cropsize=224, SRS: NORM_RESOLUTION=256
# 공식 가중치 사용 시 학습된 해상도에 맞춰야 함
NORM_RESOLUTION = 256   # 변경 시 key 길이도 달라짐

# ─── 사전 난독화 ──────────────────────────────────────────────────
DEFAULT_OBFUSCATOR = 'blur'
BLUR_KERNEL_SIZE   = 61
BLUR_SIGMA         = 21.0     # 원본 hybridAll: Blur(61, 9, 21)
BLUR_SIGMA_MIN     = 9.0      # 원본 hybridAll blur sigma_min
PIXELATE_BLOCK     = 20       # 원본 hybridAll: Pixelate(20)
MEDIAN_KERNEL      = 23       # 원본 hybridAll: MedianBlur(23)

# ─── 검출기 ───────────────────────────────────────────────────────
DETECTOR_CONF_THRESHOLD = 0.25
DETECTOR_NMS_IOU        = 0.4
FACE_MARGIN             = 0.10

# ─── 학습 하이퍼파라미터 (SRS §7 / 원본 config 기준) ─────────────
lr           = 0.00001
batch_size   = 6
weight_decay = 1e-5
init_scale   = 0.01
TRIPLET_MARGIN         = 1.2
LAMBDA_RECONSTRUCTION  = 5
LAMBDA_GUIDE           = 1
LAMBDA_LOW_FREQUENCY   = 1

SAVE_IMAGE_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 5000

# ─── 사전학습 가중치 파일명 ───────────────────────────────────────
CHECKPOINT_ID = "hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000"

# ─── KeyGen (PBKDF2) ── NFR-SEC-2 경고 ───────────────────────────
# !! salt=1, count=10 은 논문의 "demonstration only" 값 !!
# 운영 배포 시 임의 salt + OWASP 권고 반복 수(≥600000)로 교체할 것
KEY_SALT  = 1
KEY_COUNT = 10

# ─── 기타 ─────────────────────────────────────────────────────────
debug = False
recognizer = 'AdaFaceIR100'

# ─── 통합 시스템 설정 ────────────────────────────────────────────
# 복원 비밀번호 (데모용 — 실제 배포 시 환경변수로 교체)
DEMO_PASSWORD = "forensic2026"

# INN 가중치 경로 (None이면 랜덤 초기화 — 형태 확인용)
# 실제 가중치가 있으면 아래 경로 지정:
# INN_CHECKPOINT = "checkpoints/hybridAll_inv3_...pth"
INN_CHECKPOINT = f"checkpoints/{CHECKPOINT_ID}.pth"

# ArcFace 코사인 유사도 임계값
MATCH_THRESHOLD = 0.45

# 카메라 종류: "webcam"(cv2.VideoCapture) 또는 "realsense"(pyrealsense2)
# RealSense D455도 UVC 장치라 OpenCV(webcam)로 컬러 스트림을 받을 수 있음.
CAMERA_TYPE = "webcam"

# 카메라 인덱스. RealSense는 여러 /dev/videoN 중 컬러 노드를 골라야 함.
# scan_camera.py 로 어느 인덱스가 컬러 영상을 주는지 확인 후 지정.
CAMERA_INDEX = 0

# RealSense(pyrealsense2 사용 시) 컬러 스트림 설정
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30

# 카메라 장치 이름 (None이면 정수 인덱스 자동 탐색)
# Windows DSHOW: "video=<장치이름>" 형식으로 열림
CAMERA_DEVICE_NAME = "Logi C310 HD WebCam"

# 카메라 실패 시 폴백 동영상 (mp4 등). None이면 "Reconnecting" 재시도.
# 카메라 인식에만 집중 → 폴백 영상 사용 안 함.
VIDEO_FALLBACK = None
VIDEO_FALLBACK_FPS = 25   # 재생 속도 (원본 영상 fps에 맞춤)

# True이면 카메라 탐색을 건너뛰고 무조건 VIDEO_FALLBACK 영상을 사용.
# (이 PC 카메라가 검은 프레임만 줄 때 demo.mp4로 강제하는 용도)
FORCE_VIDEO = False

# PSF 녹화 간격(초). 작을수록 복원 영상이 부드럽지만 디스크 사용 증가.
RECORD_INTERVAL = 1

# 청크 길이(분). 이 시간 단위로 새 청크 폴더가 생성됨.
# 데모 안정성: 짧게(1분)면 청크당 프레임 수가 적어 복원이 빠름.
CHUNK_MINUTES = 1

# 복원 영상 재생 fps
RESTORE_VIDEO_FPS = 5

# ffmpeg 실행 파일 경로 (None이면 PATH에서 탐색).
# 브라우저 호환 H.264 변환에 사용. winget 설치 시 PATH에 없을 수 있어 직접 지정.
FFMPEG_PATH = r"C:\Users\HOSEO\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
