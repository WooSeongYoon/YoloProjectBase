# YoloProjectBase
Yolo 모델의 학습 기초

## 사전 요구사항
- pip install ultralytics
- pip install wandb

## 프로젝트 구조
YoloProjectBase/   
├── Dataset/                # 데이터셋   
│   ├─ data.yaml/                # 데이터셋 정의   
│   ├── img/   
│   │   ├── train/               # 학습   
│   │   ├── val/                 # 검증   
│   │   └── test/                # 테스트   
│   └── label/   
│       ├── train/               # 학습   
│       ├── val/                 # 검증   
│       └── test/                # 테스트   
│   
├── trainYolo.py/                # 학습 실행 코드   
│   
├── yolov8m.pt/                  # YOLO pt파일   
│   
└── docker-compose.yml            # Docker 컨테이너 구성 파일   
