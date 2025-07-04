  YOLOv8을 이용한 Floc 측정 웹 애플리케이션


  이 프로젝트는 YOLOv8 객체 탐지 모델을 사용하여 Floc(플록)을 측정하고, 그 결과를 웹 인터페이스를 통해 보여주는 애플리케이션입니다.

  주요 기능


   - 웹 인터페이스: 사용자가 이미지를 업로드하고 Floc 측정 결과를 확인할 수 있습니다.
   - Floc 측정: 업로드된 이미지에서 Floc을 탐지하고, 관련 수치(크기, 개수 등)를 계산합니다.
   - 모델 학습: train_yolov8.py 스크립트를 통해 커스텀 데이터셋으로 YOLOv8 모델을 학습시킬 수 있습니다.

  기술 스택


   - 언어: Python
   - 웹 프레임워크: Flask (또는 FastAPI 등 app.py에서 사용한 프레임워크)
   - ML/DL 라이브러리: PyTorch, Ultralytics (YOLOv8), OpenCV

  프로젝트 구조
   1 .
   2 ├── app.py                  # 웹 애플리케이션 실행 파일
   3 ├── measurement_floc.py     # Floc 측정 로직을 담은 모듈
   4 ├── train_yolov8.py         # YOLOv8 모델 학습 스크립트
   5 ├── templates/
   6 │   └── index.html          # 메인 웹 페이지 HTML
   7 └── requirements.txt        # (생성 필요) Python 패키지 의존성 목록


  설치 및 실행 방법

  1. 저장소 복제


   1 git clone <저장소 URL>
   2 cd <프로젝트 폴더>



  2. 가상 환경 생성 및 활성화


   1 # Windows
   2 python -m venv venv
   3 venv\Scripts\activate


  3. 의존성 설치

  먼저 requirements.txt 파일이 없다면 생성해야 합니다. 아래는 예상되는 패키지 목록입니다.


   1 # requirements.txt
   2 flask
   3 ultralytics
   4 torch
   5 torchvision
   6 opencv-python-headless
   7 numpy


  위 내용으로 requirements.txt 파일을 만드신 후, 아래 명령어로 패키지를 설치하세요.


   1 pip install -r requirements.txt



  4. 모델 학습 (선택 사항)

  커스텀 데이터셋으로 모델을 직접 학습시키려면 아래 명령어를 실행하세요. (데이터셋 경로 등은 train_yolov8.py 내부에서 설정해야 할 수 있습니다.)



   1 python train_yolov8.py


  5. 웹 애플리케이션 실행


   1 python app.py


  실행 후, 웹 브라우저에서 http://127.0.0.1:5000 주소로 접속하세요.

  ---
