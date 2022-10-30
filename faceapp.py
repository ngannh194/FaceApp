from PyQt5 import QtWidgets, uic, QtCore, QtGui
import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

#https://github.com/Furkan-Gulsen/face-classification

from keras.models import load_model
import numpy as np
import cv2

root= 'face-classification/'

genderModelPath = root + 'models/genderModel_VGG16.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

genders = {
    0: {
        "label": "Female",
        "color": (245, 215, 130)
    },
    1: {
        "label": "Male",
        "color": (148, 181, 192)
    },
}

# pre-trained model
modelFile = root + "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = root + "faceDetection/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detectFacesWithDNN(frame, gender= True):
    # A neural network that really supports the input value
    size = (300, 300)

    # After executing the average reduction, the image needs to be scaled
    scalefactor = 1.0

    # These are our mean subtraction values. They can be a 3-tuple of the RGB means or
    # they can be a single value in which case the supplied value is subtracted from every
    # channel of the image.
    swapRB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    resizedFrame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
    net.setInput(blob)
    dnnFaces = net.forward()
    result= []
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.4:
            box = dnnFaces[0, 0, i, 3:7] * np.array(
                [width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            if not gender:
                result.append(((x, y, x1, y1), None))
                continue
            
            # gender prediction
            resized = frame[y - 20:y1 + 30, x - 10:x1 + 10]
            try:
                frame_resize = cv2.resize(resized, genderTargetSize)
            except:
                continue
            
            frame_resize = frame_resize.astype("float32")
            frame_scaled = frame_resize / 255.0
            frame_reshape = np.reshape(frame_scaled, (1, 100, 100, 3))
            frame_vstack = np.vstack([frame_reshape])
            gender_prediction = genderClassifier.predict(frame_vstack)
            gender_probability = np.max(gender_prediction)
#             color = (255, 255, 255)
            if (gender_probability > 0.4):

                gender_label = np.argmax(gender_prediction)
#                 gender_result = genders[gender_label]["label"]
#                 color = genders[gender_label]["color"]
#                 cv2.rectangle(frame, (x + 20, y1 + 20), (x + 130, y1 + 55),
#                               color, -1)
#                 cv2.line(frame, (x, y1), (x + 20, y1 + 20), color, thickness=2)
#                 cv2.putText(frame, gender_result, (x + 25, y1 + 45),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
#                             cv2.LINE_AA)
#                 cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                result.append(((x, y, x1, y1), gender_label))
            else:
                result.append(((x, y, x1, y1), None)) #Ko nhận được giới tính
#                 cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                
    return result

# https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
import traceback, sys

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
            
import cv2
import time
from datetime import datetime

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        self.initUI()
        self.connectUI()
        
    def initUI(self):
        uic.loadUi('main_camera.ui', self)
        
        self.labelVideo = QLabel(self.widgetVideo)
        self.labelMask = QLabel(self.widgetVideo)
        
        self.labelVideo.setGeometry(self.widgetVideo.geometry())
        self.labelMask.setGeometry(self.widgetVideo.geometry())
        self.labelMask.setStyleSheet("background-color: rgba(0,0,0,0%)")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        
        
        self.threadPool= QThreadPool()

        self.timer = QTimer()
        self.frameRate = 120
        worker1= Worker(self.start)
        self.threadPool.start(worker1)
        
#         haar_file = 'haarcascade_frontalface_default.xml'
#         self.faceCascade = cv2.CascadeClassifier(haar_file)
        self.frameRateFace = 120
#         self.startDetectFace()
        self.DECTECTGENDER= self.pushButtonGender.isChecked()
        
    def connectUI(self):
        self.pushButtonDetectFace.clicked.connect(self.runDetectFace)
        self.pushButtonGender.clicked.connect(self.runDetectGender)
        self.pushButtonScreenShot.clicked.connect(self.screenshot)
    
    def screenshot(self):
        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.widgetVideo.winId() )
        now = datetime.now()
        ntr= now.strftime("%m_%d_%Y_%H_%M_%S")
        screenshot.save('screenshot/faceapp_screenshot_%s.png'%ntr, 'png')
        QMessageBox.about(self,"Thông báo!", "Chụp thành công!")
    
    def runDetectGender(self):
        self.DECTECTGENDER= self.pushButtonGender.isChecked()
    
    def runDetectFace(self):
        self.DECTECTFACE= self.pushButtonDetectFace.isChecked()
        worker2= Worker(self.startDetectFace)
        self.threadPool.start(worker2)
        
    def start(self):
        while True:
            rate = 1.0 / self.frameRate
            time.sleep(rate)
            ret, self.frame = self.cap.read() ##self.frame để lưu lại frame từ camera để detect nếu cần
            if ret == False:
                break
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.labelVideo.resize(self.widgetVideo.size())
            self.labelVideo.setPixmap(pixmap.scaledToWidth(self.labelVideo.size().width()))
        
    def detectFace(self):
        faces = detectFacesWithDNN(self.frame, gender= self.DECTECTGENDER)
        pixmapMask = QPixmap(self.frame.shape[1], self.frame.shape[0])
        pixmapMask.fill(Qt.transparent)
        painter = QPainter(pixmapMask)
        color = QColor(*(0,255,0))
        
        font = painter.font()
        font.setPointSize(40)
        painter.setFont(font)
        for ((x, y, x1, y1), genderID) in faces:
            if genderID is not None:
                genderResult = genders[genderID]["label"]
                color = QColor(*genders[genderID]["color"])
            
            painter.setPen(QPen(color, 3, Qt.SolidLine))
            painter.drawRect(x,y,x1-x,y1-y)
            if genderID is not None:
                painter.drawText(x, y - 40, genderResult)
            
        painter.end()
        self.labelMask.resize(self.widgetVideo.size())
        self.labelMask.setPixmap(pixmapMask.scaledToWidth(self.labelMask.size().width()))
        
    def startDetectFace(self):
        self.labelMask.show()
        while self.DECTECTFACE:
            rate = 1.0 / self.frameRateFace
            time.sleep(rate)
            self.detectFace()
            
        self.labelMask.hide()
        
    def closeEvent(self, event):
        close = QtWidgets.QMessageBox.question(self,
                                     "Thoát",
                                     "Bạn muốn thoát ra chứ?",
                                     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if close == QtWidgets.QMessageBox.Yes:
            self.cap.release()
            event.accept()
        else:
            event.ignore()
            
            
app = QtWidgets.QApplication(sys.argv)
ui = MainUI()
ui.show()
sys.exit(app.exec())