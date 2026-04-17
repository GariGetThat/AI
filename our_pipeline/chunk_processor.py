import torch 
import cv2
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

class ChunkProcessor:
    def __init__(self, model_cfg, checkpoint, fps=25, chunk_seconds = 15):
        # build_sam2_video_predictor 가져다가 사용하기
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cpu")
        self.fps = fps
        self.chunk_size = fps * chunk_seconds # 375 프레임

    # chunk frame만 읽고 SAM2 형식으로 반환
    # video_path = 영상 경로, start_frame ~ end_frame은 읽을 범위 
    def _load_chunk_frames(self, video_path, start_frame, end_frame):
        """영상에서 청크 프레임만 읽기"""

        img_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        cap = cv2.VideoCapture(video_path) # OpenCV로 영상 파일 열기, cap이 영상을 읽는 도구가 됨 

        # 영상의 재생 위치를 start_frame으로 이동
        # 처음부터 읽지 않고 원하는 프레임부터 바로 읽을 수 있다.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 

        frames = []
        # start_frame에서 end_frame까지 프레임 수만큼 반복
        # 예를 들어 0~375면 375번 반복
        for _ in range(end_frame - start_frame): 
            # 프레임 한 장 읽기 
            # ret = 성공 여부, frame = 이미지
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # SAM2 image_size로 resize
            frame = cv2.resize(frame, (self.predictor.image_size, self.predictor.image_size))
            # (H,W,C) -> (C,H,W) 텐서로 변환 + 0~1 정규화 
            frame = torch.tensor(frame).permute(2,0,1).float() /255.0
            frames.append(frame)
        
        cap.release()

        # 스택
        
        images = torch.stack(frames, dim=0)

        # mean, std 정규화 
        images -= img_mean
        images /= img_std

        return images
    
    
    def mask_to_box(self, mask):
        # 마스크에서 bounding box 추출
        mask_np = mask[0].cpu().numpy()
        if not mask_np.any():
            return None
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y1, y2 = np.where(rows)[0][[0,-1]]
        x1, x2 = np.where(cols)[0][[0,-1]]
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def process(self, video_path, targets):
        print("process 시작!")
        # video_path : 영상 파일 경로
        # targets = [{"id", "type", "start_frame", "end_frame", "box"}, ...]
        
        # 영상 전체 프레임 수, 해상도 파악
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"총 프레임 수: {total_frames}")
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        # 현재 활성 box (청크 간 전달용)
        active_boxes = {t["id"]:t["box"] for t in targets}

        # 결과 저장
        results = {}  # {abs_frame_idx: {obj_id: mask}}

        for chunk_start in range(0, total_frames, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_frames)
            print(f"청크 처리 중 : {chunk_start} - {chunk_end}")

            # 청크 프레임 로드(SAM2 형식으로 변환된 텐서)
            frames = self._load_chunk_frames(video_path, chunk_start, chunk_end)
            if len(frames) ==0 :
                break

            # with torch.inference_mode(), torch.autocast("cuda", dtype = torch.bfloat16) 
            with torch.inference_mode():
                # init_state에 프레임 배열 직접 전달
                state = self.predictor.init_state(
                    frames = frames,
                    video_height = video_height,
                    video_width = video_width,
                )

                # 이번 청크에서 활성화할 객체 등록
                active_targets = []
                for target in targets:
                    obj_id = target["id"]

                    # end_frame 지난 객체 스킵
                    if target["end_frame"] < chunk_start:
                        continue
                    # 아직 start_frame 안 된 객체 스킵
                    if target["start_frame"] > chunk_start:
                        continue
                    # 저장된 box가 없으면 스킵
                    if obj_id not in active_boxes:
                        continue

                    # 청크 내 상대 프레임 인덱스 
                    prompt_frame = max(target["start_frame"], chunk_start)- chunk_start

                    self.predictor.add_new_points_or_box(
                        inference_state = state,
                        frame_idx = prompt_frame,
                        obj_id = obj_id,
                        box = active_boxes[obj_id],
                    )
                    active_targets.append(target)
                
                if not active_targets:
                    self.predictor.reset_state(state)
                    del state
                    torch.cuda.empty_cache()
                    continue

                # propagate
                last_boxes = {}
                for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(state):
                    abs_frame = chunk_start +frame_idx

                    for obj_id, mask in zip(obj_ids, masks):
                        target = next(t for t in targets if t["id"] == obj_id)

                        # end_frame 지난 객체는 box 저장 안 함
                        if abs_frame > target["end_frame"]:
                            continue
                         
                        box = self.mask_to_box(mask)
                        if box is not None:
                            last_boxes[obj_id] = box

                        # 결과 저장
                        if abs_frame not in results:
                            results[abs_frame] = {}
                        results[abs_frame][obj_id] = mask.cpu().numpy()

                # 다음 청크를 위해 box 업데이트 
                for obj_id, box in last_boxes.items():
                    active_boxes[obj_id] = box

                
                # 메모리 리셋
                self.predictor.reset_state(state)
                del state
                torch.cuda.empty_cache()

        return results