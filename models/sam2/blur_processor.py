import cv2
import numpy as np

class BlurProcessor:
    def __init__(self, blur_strength=31):
        self.blur_strength = blur_strength # 블러 강도 (홀수여야 함)


    def process(self, video_path, results, targets, output_path="output_video.avi"):
        """
        video_path : 원본 영상 경로
        results : chunk_processor에서 나온 마스크 결과 
        output_path : 블러 처리된 영상 저장 경로
        """
        cap = cv2.VideoCapture(video_path)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 출력 영상 설정
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"), # avi 저장 형식
            fps,
            (int(video_width), int(video_height))  # int로 명시적 변환
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read() # 한 프레임씩 읽기
            if not ret:
                break

            # 이 프레임에 마스크가 있으면 블러 처리
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    # targets 설정에서 해당 객체 정보(type, box 등)을 가져옴
                    target = next((t for t in targets if t["id"] == obj_id), None)
                    if target is None:
                        continue
                    
                    if target["type"] == "face":
                        # SAM2 마스크로 블러
                        binary_mask = (mask[0] > 0.8).astype(np.uint8)
                        binary_mask_3ch = np.stack([binary_mask]*3, axis=-1)
                        blurred = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)
                        frame = np.where(binary_mask_3ch == 1, blurred, frame).astype(np.uint8)
                    
                    else:
                        # box 직접 블러
                        x1, y1, x2, y2 = target["box"]
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (self.blur_strength, self.blur_strength), 0)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"블러 처리 완료! 저장 경로: {output_path}")