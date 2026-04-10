import cv2
import numpy as np

# 테스트 영상 생성 (5초, 25fps, 640x480)
out = cv2.VideoWriter(
    "test_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    25,
    (640, 480)
)

for i in range(125):  # 25fps * 5초 = 125프레임
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out.write(frame)

out.release()
print("테스트 영상 생성 완료!")