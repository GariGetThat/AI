from our_pipeline.chunk_processor import ChunkProcessor

processor = ChunkProcessor(
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
)

targets = [
    {
        "id": "face_001",
        "type": "face",
        "start_frame": 0,
        "end_frame": 750,
        "box": [100, 200, 300, 600]
    },
    {
        "id": "object_001",
        "type": "object",
        "label": "번호판",
        "start_frame": 0,
        "end_frame": 600,
        "box": [500, 300, 620, 340]
    },
]

results = processor.process("video.mp4", targets)