from our_pipeline.chunk_processor import ChunkProcessor
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="영상 파일 경로")
parser.add_argument("--targets", type=str, required=True, help="targets JSON 파일 경로")
args = parser.parse_args()

# targets JSON 파일 읽기
with open(args.targets, "r") as f:
    targets = json.load(f)

processor = ChunkProcessor(
    model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml", # 모델 구조 설정 파일
    checkpoint="checkpoints/sam2.1_hiera_small.pt", # 학습된 가중치 파일 
)

results = processor.process("video.mp4", targets)

np.save("results.npy", results)

# 실행 시 
# python our_pipeline/main.py --video video.mp4 --targets our_pipeline/targets.json

# results 전달 받은 후 어떻게 처리할지는 아직 코드 작성 x(OpenCV 연결하는 부분)