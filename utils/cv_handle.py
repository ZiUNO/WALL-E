# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/11/25 13:19
"""

import cv2
import os


def jpg_cv():
    path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['data', 'tmp'])
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = frame[60:420, 140:500]
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path + os.sep + 'image.jpeg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
