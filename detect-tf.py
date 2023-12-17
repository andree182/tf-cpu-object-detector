#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import sys
import multiprocessing

minp = 0.7
numcpus = multiprocessing.cpu_count()

wanted = ["person", "cat", "dog", "pizza"]
signatures = [x.split(' ')[2] for x in open('labelmap.txt').read().split('\n')]
show_video = False
video_out = False
process_all_video = False

while sys.argv[1][0] == '-':
    if sys.argv[1] == '--show':
        show_video = True
    if sys.argv[1] == '--write':
        video_out = True
    if  sys.argv[1] == '--all':
        process_all_video = True

    sys.argv = [sys.argv[0]] + sys.argv[2:]

# OpenCV video capture
video_path = sys.argv[1]
name = "det-" + video_path
cap = cv2.VideoCapture(video_path)

out = None

print("PROCESSING: " + video_path)

class detector:
    def __init__(self):
       self.model_path = 'cpu_model.tflite'  # Replace with the path to your .tflite file
       self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads = numcpus - 2)

       self.interpreter.allocate_tensors()

       self.tensor_input_details = self.interpreter.get_input_details()
       self.tensor_output_details = self.interpreter.get_output_details()

    def detect_raw(self, tensor_input):
       self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
       self.interpreter.invoke()

       boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
       class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
       scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
       count = int(
          self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0]
       )

       detections = np.zeros((20, 6), np.float32)

       for i in range(count):
          if scores[i] < minp or i == 20:
             break
          detections[i] = [
             class_ids[i],
             float(scores[i]),
             boxes[i][0],
             boxes[i][1],
             boxes[i][2],
             boxes[i][3],
          ]

       return detections

det = detector()

detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
       break
    if video_out and out == None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    input_shape = det.tensor_input_details[0]['shape']
    # Preprocess the frame
    input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(input_data, axis=0)

    dets = det.detect_raw(input_data)
    
    for d in dets:
        if d[1] < minp:
            continue
        sigstr = signatures[d[0].astype(int)]
        prob = (d[1] * 100).astype(int)
        if not sigstr in wanted:
            continue
        detected = True
        print('DETECTED: {}|{}'.format(sigstr, prob))
        
        x1, y1, x2, y2 = d[3] * frame.shape[1], d[2] * frame.shape[0], d[5] * frame.shape[1], d[4] * \
                            frame.shape[0]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, '{}|{}'.format(sigstr, prob), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    if show_video:
        cv2.imshow('Signature Detection', frame)
    if video_out:
        out.write(frame)

    if detected and not process_all_video:
        break

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# Release the video capture and close the OpenCV window
if out != None:
    out.release()
cap.release()
cv2.destroyAllWindows()

sys.exit(0 if detected else 1)
