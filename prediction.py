import cv2
import argparse
from load import FacialExpressionModel
import numpy as np

# Use OpenCV's built-in haarcascade path to avoid missing local file issues
facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self, camera_index: int = 0):
        self.video = cv2.VideoCapture(camera_index)
    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()
    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        ok, fr = self.video.read()
        if not ok or fr is None:
            print("Error: Failed to capture frame")
            return None
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            if fc.size == 0:
                continue
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi)
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return fr

def gen(camera: VideoCamera, window_title: str = 'Facial Expression Recognition', quit_key: str = 'q'):
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run webcam facial expression recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--title', type=str, default='Facial Expression Recognition', help='Window title')
    parser.add_argument('--quit', type=str, default='q', help='Quit key')
    args = parser.parse_args()
    gen(VideoCamera(args.camera), window_title=args.title, quit_key=args.quit)