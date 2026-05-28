import cv2
import json

cap = cv2.VideoCapture('/home/undergraduate/20221373_YY/capstone/AI/demo_cap.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('debug_boxes.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

with open('our_pipeline/targets.json') as f:
    targets = json.load(f)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for target in targets:
        if target['start_frame'] <= frame_idx <= target['end_frame']:
            x1, y1, x2, y2 = target['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, target['id'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print('완료!')