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

# Hailo-8L 가속: SCRFD 탐지 + ArcFace 인식을 Hailo에서 실행.
# True여도 hailo_platform이 없으면 자동으로 insightface(CPU)로 폴백.
# ⚠️ Hailo ArcFace는 임베딩 공간이 달라 전환 시 재등록 필요.
USE_HAILO = True
SCRFD_HEF_PATH = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
ARCFACE_HEF_PATH = "/usr/share/hailo-models/arcface_mobilefacenet.hef"
HAILO_DET_THRESH = 0.5

# True면 내부인(등록 사원)도 익명화 (전원 보호, 권한자만 복원).
# False면 외부인만 익명화하고 내부인은 신원 표시.
ANONYMIZE_ALL = True

# 실시간 화면 익명화 방식:
#   "mosaic" — 가벼운 모자이크(실시간 부드러움). INN 보호본은 녹화 시 별도 생성.
#   "inn"    — 실시간 화면도 INN 보호본(느림, 라즈베리파이 비권장).
# 어느 쪽이든 녹화/복원은 INN으로 동작 → 복원하면 원본이 나옴.
REALTIME_ANON = "mosaic"

# 처리·저장 fps 상한(초당 장수).
# INN protect가 라즈베리파이 CPU에서 ~2.3초/프레임(=0.4fps)이라, 이보다
# 크게 잡으면 pending 큐가 쌓여 청크가 완성되지 않는다. INN 속도에 맞춰
# 낮게(1 이하) 두어야 실시간에 근접해 청크가 제때 완성된다. 대신 복원
# 영상은 이 fps만큼만 촘촘(낮으면 뚝뚝).
PROCESS_MAX_FPS = 1

# (구) 프레임 스킵. 시간 기반 PROCESS_MAX_FPS를 쓰므로 1로 둠.
PROCESS_EVERY_N = 1

# recorder가 뒤처질 때 원본 프레임을 쌓아두는 큐 최대 크기(장).
# 클수록 프레임을 덜 버리지만 메모리↑ (JPEG 압축 저장, 장당 ~50KB).
FRAME_QUEUE_MAX = 3000

# 처리 전 프레임 가로 해상도 축소(px). 0이면 원본. 탐지 속도↑.
PROCESS_WIDTH = 0

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

# PSF 녹화 간격(초). 0이면 INN이 처리 가능한 최대 속도로 쉬지 않고 저장
# (가장 부드럽지만 CPU 최대 사용). 실제 저장 fps는 로그로 확인.
RECORD_INTERVAL = 0

# 청크 길이(분). INN이 느려 완성에 시간이 걸리므로 데모는 1분 권장.
# 저장 계층: recordings/월/일/오전오후/시/N분청크
CHUNK_MINUTES = 1

# 복원 영상 출력 fps (부드러움용). 실제 영상 길이는 프레임 타임스탬프로 맞춰짐.
RESTORE_VIDEO_FPS = 15

# Modal 서버리스 GPU 복원 엔드포인트. None이면 GPU 복원 버튼 비활성.
MODAL_RESTORE_URL = "https://yena07--securefacerx-restore-restore.modal.run"

# ffmpeg 실행 파일 경로 (None이면 PATH에서 탐색).
# 브라우저 호환 H.264 변환에 사용. winget 설치 시 PATH에 없을 수 있어 직접 지정.
FFMPEG_PATH = r"C:\Users\HOSEO\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
