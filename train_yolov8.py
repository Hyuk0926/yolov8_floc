import torch
from ultralytics import YOLO
import argparse
import os

def train_yolo(data_yaml, epochs, patience, batch_size, imgsz, project_name, run_name):
    """
    YOLOv8 모델을 학습하고 과적합 방지 기능을 적용합니다.

    Args:
        data_yaml (str): data.yaml 파일 경로
        epochs (int): 최대 학습 에포크 수
        patience (int): 조기 종료를 위한 대기 에포크 수
        batch_size (int): 배치 사이즈
        imgsz (int): 학습 이미지 크기
        project_name (str): 학습 결과를 저장할 프로젝트 폴더 이름
        run_name (str): 현재 학습 실행의 이름
    """
    # 사전 학습된 YOLOv8 모델 로드 (yolov8n.pt는 가장 작고 빠른 모델)
    # 정확도가 더 필요하다면 'yolov8s.pt' 또는 'yolov8m.pt' 등을 사용할 수 있습니다.
    model = YOLO('yolov8n.pt')

    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # model.train() 함수를 사용하여 모델 학습
    # 이 함수는 데이터 로딩, 데이터 증강, 학습, 검증, 모델 저장을 모두 자동으로 처리합니다.
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        patience=patience,  # 검증 손실이 'patience' 에포크 동안 개선되지 않으면 학습을 조기 종료합니다.
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_name, # 결과를 저장할 상위 폴더
        name=run_name         # 이번 실행 결과를 저장할 하위 폴더
    )

    print("Training complete.")
    # 학습 결과가 저장된 경로를 출력합니다.
    # ultralytics 라이브러리의 최신 버전에서는 results 객체가 save_dir 속성을 가지고 있지 않을 수 있습니다.
    # 이 경우, project/name 경로를 직접 확인해야 합니다.
    save_dir = os.path.join(project_name, run_name)
    print(f"Results saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with overfitting prevention.")
    
    # 필수 인자
    parser.add_argument('--data', type=str, required=True, help='Path to the data.yaml file.')
    
    # 선택 인자 (기본값 설정)
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs to train for.')
    parser.add_argument('--patience', type=int, default=50, help='Epochs to wait for no observable improvement before stopping training.')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training. Adjust based on your GPU memory.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--project', type=str, default='runs/train', help='Directory to save training runs.')
    parser.add_argument('--name', type=str, default='exp', help='Name for the specific training run.')

    args = parser.parse_args()

    train_yolo(args.data, args.epochs, args.patience, args.batch, args.imgsz, args.project, args.name)