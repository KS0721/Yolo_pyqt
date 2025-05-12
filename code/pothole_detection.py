import cv2
import os
import sys
import datetime
import pymysql
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from ultralytics import YOLO  # YOLOv8 모델 로드
import numpy as np

class PotholeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pothole Detection")
        self.setGeometry(100, 100, 800, 600)

        # YOLOv5 모델 로드
        self.model = YOLO("pothole.pt")  # 학습된 모델 경로

        # 웹캠 설정
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "오류", "카메라를 열 수 없습니다.")
            sys.exit()

        # 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        """웹캠 프레임을 읽고 YOLO 모델로 분석"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # YOLO 모델로 분석
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # 클래스 ID
                conf = box.conf[0]  # 신뢰도
                if conf > 0.5:  # 신뢰도 조건
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 박스
                    cv2.putText(frame, f"Pothole {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # MySQL 데이터베이스에 저장
                    self.save_to_database(cls, conf, x1, y1, x2, y2)

        # 프레임을 화면에 표시
        cv2.imshow("Pothole Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def save_to_database(self, cls, conf, x1, y1, x2, y2):
        """MySQL 데이터베이스에 결과 저장"""
        try:
            connection = pymysql.connect(
                host="localhost",
                user="root",
                password="password",
                database="pothole_db"
            )
            cursor = connection.cursor()
            query = """
                INSERT INTO detections (class_id, confidence, x1, y1, x2, y2, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(query, (cls, conf, x1, y1, x2, y2, timestamp))
            connection.commit()
            cursor.close()
            connection.close()
        except Exception as e:
            print(f"데이터베이스 저장 실패: {e}")

    def closeEvent(self, event):
        """종료 시 자원 해제"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

# 애플리케이션 실행
if __name__ == "__main__":
    app = QApplication([])
    window = PotholeDetectionApp()
    window.show()
    app.exec()