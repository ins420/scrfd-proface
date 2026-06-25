"""
가장 기본적인 OpenCV 카메라 출력 (velog 강좌 형태).
서버와 무관하게 이 PC에서 카메라가 되는지 창으로 직접 확인.

사용:
  python test_camera_basic.py        # 카메라 0
  python test_camera_basic.py 1      # 카메라 1
  q 키로 종료
"""
import sys
import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(idx)

if not cap.isOpened():
    print(f"카메라 {idx}를 열 수 없습니다.")
    sys.exit()

print(f"카메라 {idx} 열림. q 키로 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
