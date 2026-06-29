"""
카메라 인덱스 0~8을 OpenCV로 스캔해 어떤 게 컬러 영상을 주는지 확인.
RealSense는 /dev/video0~N 중 일부만 컬러(나머지는 depth/IR)이므로
실제 컬러가 나오는 인덱스를 찾아 config.CAMERA_INDEX 에 지정한다.

사용:
  python scan_camera.py
"""
import cv2

print("카메라 인덱스 스캔 (0~8)...\n")

found = []
for idx in range(9):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        continue

    ok_count = 0
    bright = 0.0
    shape = None
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_count += 1
            bright = float(frame.mean())
            shape = frame.shape

    if ok_count > 0:
        note = ""
        if bright < 3:
            note = "  ⚠️ 검은/빈 프레임"
        elif shape and len(shape) == 3 and shape[2] == 3:
            note = "  ✅ 컬러 영상"
            found.append(idx)
        print(f"idx={idx}: 열림  읽기 {ok_count}/10  shape={shape}  밝기={bright:.1f}{note}")
    else:
        print(f"idx={idx}: 열렸지만 읽기 실패")
    cap.release()

print("\n" + "=" * 50)
if found:
    print(f"컬러 영상이 나오는 인덱스: {found}")
    print(f"→ config.py 의 CAMERA_INDEX = {found[0]} 로 설정하세요.")
else:
    print("컬러 영상을 주는 카메라를 찾지 못했습니다.")
    print("RealSense가 /dev/video* 로 잡히는지 'ls /dev/video*' 로 확인하세요.")
