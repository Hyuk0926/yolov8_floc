from flask import Flask, render_template, jsonify, url_for
import pandas as pd
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- 설정 변수 ---
# 나중에 measurement_floc.py와 연동할 때 이 부분을 동적으로 설정할 수 있습니다.
# 지금은 measurement_floc.py의 기본 출력 경로를 사용합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'analysis_results')
CSV_FILE_PATH = os.path.join(RESULTS_DIR, 'sedimentation_analysis.csv')
IMAGES_DIR = os.path.join(RESULTS_DIR, 'visualized_images')

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    # HTML 템플릿을 렌더링하여 반환합니다.
    # 필요한 데이터는 아래 API 엔드포인트를 통해 JavaScript에서 비동기적으로 요청합니다.
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """CSV 데이터를 읽어 JSON 형식으로 반환하는 API 엔드포인트입니다."""
    try:
        if not os.path.exists(CSV_FILE_PATH):
            logging.error(f"CSV file not found at: {CSV_FILE_PATH}")
            return jsonify({"error": "CSV file not found. Please run the analysis first."}), 404

        df = pd.read_csv(CSV_FILE_PATH)
        
        # Chart.js가 사용하기 편한 형태로 데이터를 가공합니다.
        chart_data = {
            'labels': df['Image'].tolist(),
            'estimated_height': df['Estimated_Height'].tolist(),
            'smoothed_height': df['Smoothed_Height'].tolist(),
            'floc_count': df['Floc_Count'].tolist(),
            'turbidity': df['Turbidity'].tolist(),
            'small_flocs': df['Small_Flocs'].tolist(),
            'medium_flocs': df['Medium_Flocs'].tolist(),
            'large_flocs': df['Large_Flocs'].tolist()
        }
        return jsonify(chart_data)

    except Exception as e:
        logging.error(f"Error reading or processing CSV file: {e}")
        return jsonify({"error": "Failed to process data."}), 500

@app.route('/api/images')
def get_images():
    """분석된 이미지 파일 목록을 JSON 형식으로 반환하는 API 엔드포인트입니다."""
    try:
        if not os.path.exists(IMAGES_DIR):
            logging.error(f"Images directory not found at: {IMAGES_DIR}")
            return jsonify({"error": "Visualized images directory not found."}), 404
            
        image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.startswith('visualized_') and (f.endswith('.png') or f.endswith('.jpg'))])
        
        # 각 이미지에 대한 정보 (URL, 원본 파일명)를 리스트로 구성
        image_data = []
        for filename in image_files:
            original_filename = filename.replace('visualized_', '')
            image_data.append({
                'url': url_for('static', filename=f'visualized_images/{filename}', _external=True),
                'original_filename': original_filename
            })

        # app.py와 동일한 위치에 static 폴더를 만들고 그 안에 visualized_images 폴더를 링크하거나 복사해야 합니다.
        # Flask는 기본적으로 'static' 폴더에서 정적 파일을 찾습니다.
        # 이를 자동화하기 위해 심볼릭 링크를 생성합니다.
        static_images_path = os.path.join(BASE_DIR, 'static', 'visualized_images')
        if not os.path.exists(static_images_path):
            os.makedirs(os.path.dirname(static_images_path), exist_ok=True)
            try:
                # 심볼릭 링크 생성 (Windows에서는 관리자 권한 필요할 수 있음)
                os.symlink(IMAGES_DIR, static_images_path)
                logging.info(f"Created symlink from {IMAGES_DIR} to {static_images_path}")
            except OSError as e:
                 logging.warning(f"Could not create symlink: {e}. You may need to run as administrator or copy the folder manually.")
                 # 심볼릭 링크 생성 실패 시, 사용자에게 수동 복사 안내
                 return jsonify({"error": f"Could not access image files. Please copy '{IMAGES_DIR}' to '{os.path.join(BASE_DIR, 'static')}'."}), 500
            except Exception as e:
                logging.error(f"An unexpected error occurred while creating symlink: {e}")
                return jsonify({"error": "An unexpected error occurred while preparing image files."}), 500


        return jsonify(image_data)

    except Exception as e:
        logging.error(f"Error listing image files: {e}")
        return jsonify({"error": "Failed to list image files."}), 500


if __name__ == '__main__':
    # debug=True로 설정하면 코드 변경 시 서버가 자동으로 재시작됩니다.
    # use_reloader=False는 초기 실행 시 중복 로딩을 방지합니다.
    app.run(debug=True, port=5001, use_reloader=False)
