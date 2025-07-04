# -*- coding: utf-8 -*-
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import os
import csv
import logging
import argparse
from scipy.signal import savgol_filter
import cv2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_turbidity(image: Image.Image) -> float:
    """이미지의 평균 픽셀 강도를 기반으로 탁도를 계산합니다."""
    image_gray = np.array(image.convert('L'))
    avg_intensity = np.mean(image_gray)
    turbidity = 255 - avg_intensity
    return turbidity

def visualize_sedimentation(image: Image.Image, output_path: str, estimated_height: float, smoothed_height: float, current_boxes: np.ndarray, floc_sizes: list, size_thresholds: tuple):
    """탐지된 Floc과 추정된 침전 높이를 시각화하여 이미지로 저장합니다."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    small_thresh, medium_thresh = size_thresholds

    for box, area in zip(current_boxes, floc_sizes):
        if area < small_thresh:
            color = 'blue'
        elif small_thresh <= area < medium_thresh:
            color = 'yellow'
        else:
            color = 'red'
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)

    if estimated_height is not None:
        line_y = h - estimated_height
        draw.line([(0, line_y), (w, line_y)], fill='blue', width=2)
        draw.text((10, line_y - 10), "Estimated Height", fill='blue')

    if smoothed_height is not None:
        smoothed_line_y = h - smoothed_height
        draw.line([(0, smoothed_line_y), (w, smoothed_line_y)], fill='red', width=3)
        draw.text((10, smoothed_line_y - 10), "Smoothed Height", fill='red')

    try:
        image.save(output_path)
    except Exception as e:
        logging.error(f"Error saving image {output_path}: {e}")

def estimate_height_from_yolo(current_boxes: np.ndarray, image_height: int, percentile: int = 15) -> float | None:
    """YOLO 결과에서 Floc들의 y좌표 백분위수를 이용해 침전 높이를 추정합니다."""
    if len(current_boxes) == 0:
        return None
    y_coords = [box[1] for box in current_boxes]
    yolo_y = np.percentile(y_coords, percentile)
    return image_height - yolo_y

def estimate_height_intensity_profile(image: Image.Image, window_length: int = 101, polyorder: int = 2) -> float:
    """이미지의 수직 픽셀 강도 프로파일을 분석하여 침전 높이를 추정합니다."""
    image_gray = np.array(image.convert('L'))
    vertical_profile = np.mean(image_gray, axis=1)
    smoothed_profile = savgol_filter(vertical_profile, window_length=window_length, polyorder=polyorder)
    gradient = np.gradient(smoothed_profile)
    sedimentation_y = np.argmax(gradient)
    return image_gray.shape[0] - sedimentation_y

class HeightSmoother:
    """Savitzky-Golay 필터를 사용하여 높이 추정치를 부드럽게 만듭니다."""
    def __init__(self, window_size=5, poly_order=2):
        self.heights = []
        self.window_size = max(window_size, 3) # window_size는 poly_order보다 커야 함
        self.poly_order = min(poly_order, self.window_size - 1)

    def smooth(self, height: float | None) -> float | None:
        if height is None:
            return self.heights[-1] if self.heights else None
        
        self.heights.append(height)
        if len(self.heights) < self.window_size:
            return np.mean(self.heights)
        
        smoothed = savgol_filter(self.heights[-self.window_size:], self.window_size, self.poly_order)
        return smoothed[-1]

def preprocess_image(image: Image.Image) -> Image.Image:
    """CLAHE를 적용하여 이미지의 대비를 향상시킵니다."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe_cv = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(image_clahe_cv, cv2.COLOR_BGR2RGB))

def analyze_images(model_path, images_dir, output_csv, output_images_dir, imgsz, conf, iou, **kwargs):
    """
    이미지 폴더를 분석하여 Floc 정보와 침전 속도를 CSV에 저장하고 시각화합니다.
    """
    model = YOLO(model_path)
    height_smoother = HeightSmoother(window_size=kwargs['smoother_window_size'])
    
    # Floc 크기 분류 기준
    size_thresholds = (kwargs['small_floc_thresh'], kwargs['medium_floc_thresh'])

    # CSV 파일 준비
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Estimated_Height", "Smoothed_Height", "Floc_Count", "Small_Flocs", "Medium_Flocs", "Large_Flocs", "Turbidity"])

        # stream=True 옵션으로 메모리를 효율적으로 사용하며 폴더 내 모든 이미지 처리
        results_generator = model.predict(source=images_dir, imgsz=imgsz, conf=conf, iou=iou, stream=True, verbose=False)

        for result in results_generator:
            image_file = os.path.basename(result.path)
            logging.info(f"Processing {image_file}...")
            
            # PIL Image로 변환
            image = Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
            
            # 전처리 (필요 시)
            preprocessed_image = preprocess_image(image)
            image_height = preprocessed_image.size[1]

            # YOLO 결과 추출
            current_boxes = result.boxes.xyxy.cpu().numpy()
            floc_count = len(current_boxes)

            # Floc 크기 계산 및 분류
            floc_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in current_boxes]
            small_count = sum(1 for area in floc_sizes if area < size_thresholds[0])
            medium_count = sum(1 for area in floc_sizes if size_thresholds[0] <= area < size_thresholds[1])
            large_count = sum(1 for area in floc_sizes if area >= size_thresholds[1])

            # 침전 높이 추정
            if floc_count > kwargs['initial_stage_threshold']:
                estimated_height = estimate_height_intensity_profile(preprocessed_image)
            else:
                estimated_height = estimate_height_from_yolo(current_boxes, image_height, percentile=kwargs['yolo_height_percentile'])

            smoothed_height = height_smoother.smooth(estimated_height)
            
            # 탁도 계산
            turbidity = calculate_turbidity(preprocessed_image)

            # CSV에 결과 저장
            writer.writerow([image_file, estimated_height, smoothed_height, floc_count, small_count, medium_count, large_count, turbidity])

            # 시각화 결과 저장
            output_image_path = os.path.join(output_images_dir, f"visualized_{image_file}")
            visualize_sedimentation(preprocessed_image, output_image_path, estimated_height, smoothed_height, current_boxes, floc_sizes, size_thresholds)
    
    logging.info(f"Analysis complete. Results saved to {output_csv} and {output_images_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze floc images and measure sedimentation speed.")
    
    # 필수 인자
    parser.add_argument('--model', type=str, required=True, help="Path to the trained YOLO model weights (e.g., best.pt).")
    parser.add_argument('--images', type=str, required=True, help="Path to the directory containing images to analyze.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory to save results (CSV and visualized images).")

    # 추론 파라미터
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for inference. Should match the training size.")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument('--iou', type=float, default=0.5, help="IOU threshold for Non-Maximum Suppression.")

    # 분석 파라미터
    parser.add_argument('--initial_stage_threshold', type=int, default=1500, help="Floc count threshold to switch height estimation method.")
    parser.add_argument('--yolo_height_percentile', type=int, default=15, help="Percentile for YOLO-based height estimation.")
    parser.add_argument('--smoother_window_size', type=int, default=5, help="Window size for Savitzky-Golay height smoother.")
    parser.add_argument('--small_floc_thresh', type=int, default=50, help="Area threshold for classifying small flocs.")
    parser.add_argument('--medium_floc_thresh', type=int, default=100, help="Area threshold for classifying medium flocs.")
    
    args = parser.parse_args()

    # 결과 저장 폴더 생성
    output_images_dir = os.path.join(args.output_dir, 'visualized_images')
    os.makedirs(output_images_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, 'sedimentation_analysis.csv')

    # **kwargs로 추가 분석 파라미터 전달
    analysis_params = {
        'initial_stage_threshold': args.initial_stage_threshold,
        'yolo_height_percentile': args.yolo_height_percentile,
        'smoother_window_size': args.smoother_window_size,
        'small_floc_thresh': args.small_floc_thresh,
        'medium_floc_thresh': args.medium_floc_thresh,
    }

    analyze_images(
        model_path=args.model,
        images_dir=args.images,
        output_csv=output_csv_path,
        output_images_dir=output_images_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        **analysis_params
    )

if __name__ == "__main__":
    main()
