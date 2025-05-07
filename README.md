# Yolo_pyqt
## 오늘은 전에 만들어둔 MainWebCam을 가지고 욜로 8을 활용해서 객체 인식을 할 수 있게 만들예정 입니다.

### 1. 전에 만들어둔 MainWebCam을 복붙해서 Juso_Yolo 파일을 만듭니다.

![image](https://github.com/user-attachments/assets/9bc4024a-244a-4d32-82fc-42b16905d096)

### 2. 가상환경을 만들고 YOLOv8을 git clone해 옵니다.
```bash
conda create -n juso_yolo python=3.9

git clone https://github.com/ultralytics/ultralytics.git
```
### 3. requirements.txt를 인스톨 합니다.
```bash
pip install -r requirements.txt

pip install ultralytics
```
## 핵심 코드 설명

### 1. YOLO 모델 로드
```
from ultralytics import YOLO
self.model = YOLO(r"D:\...path...\best.pt")
```
### 2. 웹캠 실시간 감지 및 시각화
```
results = self.model(frame)
annotated_frame = results[0].plot()
```

![image](https://github.com/user-attachments/assets/e283bea2-f946-413c-aebf-7ed778cb81aa)
