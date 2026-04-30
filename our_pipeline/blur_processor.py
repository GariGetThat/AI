import cv2
import numpy as np

class BlurProcessor:
    def __init__(self, blur_strength=51):
        self.blur_strength = blur_strength # 블러 강도 (홀수여야 함)

    def process(self, video_path, results, output_path="output_video.avi"):
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
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (int(video_width), int(video_height))  # int로 명시적 변환
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 이 프레임에 마스크가 있으면 블러 처리
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    # binary mask 변환 (True/False)
                    binary_mask = mask[0] > 0  # (H, W)

                    # 블러 처리
                    blurred = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)
                    frame[binary_mask] = blurred[binary_mask]

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"블러 처리 완료! 저장 경로: {output_path}")