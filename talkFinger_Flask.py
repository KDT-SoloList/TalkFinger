from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import cv2
import pandas as pd
from openvino.inference_engine import IECore
from collections import deque

app = Flask(__name__)

#모델 경로 적절하게 변경
model_path = "C:\\Users\\your_address\\quantized_talkFinger.xml"
ie = IECore()
net = ie.read_network(model=model_path)
exec_net = ie.load_network(network=net, device_name='CPU')
input_blob = next(iter(net.input_info))


# labels = pd.read_csv("C:\\Users\\yooji\python_AI\\openvino_project\\quantized_talkFinger\\labels.txt", header=None).values
# 라벨 경로 적절하게 변경
labels = pd.read_excel("./quantized_talkFinger/talkfinger_label50.xlsx")
labels = labels.drop("file_name", axis=1)

def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0
    return frame_normalized

# html파일들은 templates폴더에 위치해야 하고 templates폴더는 해당 파이썬 파일과 같은 위치에 있어야 함
@app.route('/')
def index():
    return render_template("index.html", data = "hey")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        video = request.files['video']
        video.save('uploaded_video.mp4')

        final_prediction = ""
        cap = cv2.VideoCapture('uploaded_video.mp4')

        if not cap.isOpened():
            print("Failed to open video")
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            predictions = []
            recent_frames = deque(maxlen=80)

            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = preprocess_frame(frame)
                prediction = exec_net.infer(inputs={input_blob: [processed_frame]})
                predictions.append(prediction)

                for i, prediction in enumerate(predictions):
                    output_tensor = next(iter(prediction.values()))
                    highest_score_index = np.argmax(output_tensor, axis=1)[0]
                    final_prediction = labels.label[highest_score_index]
                    # print(f"Frame {i + 1} - Predicted Class: {highest_score_index}")

            cap.release()
            
        return render_template("predict.html", data=final_prediction)
            # return jsonify({'result': str(outputs)})
    return redirect(url_for('index'))
    
if __name__ == '__main__':
    # cmd -> ipconfig -> IPv4 주소
    app.run(host='your_ip', port=5000, debug=True)
