import cv2
import numpy as np

out = cv2.VideoWriter(
    "test_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    25,
    (640, 480)
)

for i in range(125):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out.write(frame)

out.release()
print("테스트 영상 생성 완료!")