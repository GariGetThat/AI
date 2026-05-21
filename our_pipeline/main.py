from our_pipeline.chunk_processor import ChunkProcessor
import numpy as np
import argparse
import json
from our_pipeline.blur_processor import BlurProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="영상 파일 경로") # 처리할 영상 경로 지정 옵션
parser.add_argument("--targets", type=str, required=True, help="targets JSON 파일 경로") # 좌표 및 프레임 정보 JSON 경로 지정 옵션
args = parser.parse_args() # 터미널에 입력된 실제 인자값들을 변수에 할당

# targets JSON 파일 읽기
with open(args.targets, "r") as f:
    targets = json.load(f)

processor = ChunkProcessor(
    model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml", # 모델 구조 설정 파일
    checkpoint="checkpoints/sam2.1_hiera_small.pt", # 학습된 가중치 파일 
)

results = processor.process(args.video, targets)

# np.save("results.npy", results)

blur = BlurProcessor(blur_strength=11)
blur.process(args.video, results, targets, output_path="output_video.avi")
