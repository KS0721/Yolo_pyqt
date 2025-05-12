from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import cv2
import os
import sys
import datetime
import csv
from ultralytics import YOLO  # YOLOv8 모델
import numpy as np

# PyInstaller 대응 리소스 경로 함수
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# UI 로드
try:
    ui_path = resource_path("pyqtapp2.ui")
    Form, Window = uic.loadUiType(ui_path)
except Exception as e:
    print(f"UI 파일 로드 실패: {e}")
    sys.exit()

class MyWindow(QMainWindow, Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 버튼 연결
        self.filmBut.clicked.connect(self.capture_photo)
        self.SaveBut.clicked.connect(self.save_files)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # ✅ YOLOv8 사용자 모델 경로 설정
        self.model = YOLO(r"D:\Users\Users\Python\Juso_Yolo\runs\detect\train4\weights\best.pt")

        self.detected_classes = []  # 감지 결과 저장용 리스트

        self.start_camera()

    def start_camera(self):
        """웹캠 시작"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "카메라 오류", "카메라를 열 수 없습니다.")
            sys.exit()
        self.timer.start(30)

    def update_frame(self):
        """프레임 업데이트 및 객체 감지"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return

            results = self.model(frame)
            self.detected_classes = []  # 매 프레임마다 초기화

            annotated_frame = results[0].plot()  # 바운딩 박스 있는 프레임
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                self.detected_classes.append(f"{class_name} ({conf:.2f})")

            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.label_Cam.setPixmap(QPixmap.fromImage(q_img))

    def capture_photo(self):
        """사진 촬영"""
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "경고", "카메라가 작동 중이지 않습니다.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_file_name = f"image_{timestamp}.png"

        self.lineEdit_file.setText(self.image_file_name)

        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame)
            annotated = results[0].plot()
            cv2.imwrite(self.image_file_name, annotated)

            self.name = self.lineEdit_Name.text().strip()
            self.num = self.lineEdit_Num.text().strip()
            self.remark = self.textEdit_Remark.toPlainText().strip()

            QMessageBox.information(self, "촬영 완료", f"이미지 저장 완료: {self.image_file_name}")
        else:
            QMessageBox.critical(self, "오류", "사진 촬영 실패")

    def save_files(self):
        """CSV 저장"""
        if not hasattr(self, 'image_file_name'):
            QMessageBox.warning(self, "경고", "먼저 사진을 촬영하세요.")
            return

        csv_file_name = self.image_file_name.replace('.png', '.csv')
        with open(csv_file_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Number", "Remark", "Image File", "Detected Classes"])
            writer.writerow([
                self.name,
                self.num,
                self.remark,
                self.image_file_name,
                ', '.join(self.detected_classes)
            ])

        QMessageBox.information(self, "저장 완료", f"CSV 저장 완료: {csv_file_name}")

    def closeEvent(self, event):
        """종료 처리"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

# 실행
if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
