# import the necessary packages
import time
import imutils
import argparse
import numpy as np
import cv2
import os
from centroidtracker import CentroidTracker

# создание парсера аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="путь к входному видео")
ap.add_argument("-o", "--output", required=True,
                help="путь для выходного видео")
ap.add_argument("-y", "--yolo", required=True,
                help="путь к файлам конфигурации сети")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
                help="минимальная вероятность для фильтрации слабых обнаружений")
args = vars(ap.parse_args())

# Загрузка классовых меток датасета COCO на котором была обучена модель YOLO
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Назначение путей к файлам конфигурации нейросети и ее весам
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Загрузка модели YOLA
print("[INFO] Загрузка модели и весов YOLO...")
ct = CentroidTracker(100)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# определение общего кол-ва фреймов
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} Общее кол-во фреймов в видео".format(total))

except:
    print("ERROR")
    total = -1

while True:
    # считывание следующего фрейма из файла
    (grabbed, frame) = vs.read()

    # если фрейм не был прочитан, значит мы достигли конца файла
    if not grabbed:
        break

    # если параметры фрейма пустые, присваиваем им значения
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Создаем BLOB из входного фрейма и пропускаем его через нейросеть
    # что дает нам bounding box и связанные с ней вероятности
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # инициалиация списков детектированных объектов
    boxes = []
    confidences = []
    classIDs = []
    rects = []
    objects = None
    for output in layerOutputs:
        for detection in output:
            # достаем из детектированного объекта ID его класса и вероятность опредления этого класса
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Фильтрация объектов, которые были детектированны хуже всего
            if confidence > args["confidence"] and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                rects.append(box.astype("int"))
                (centerX, centerY, width, height) = box.astype("int")

                # нахождение левого верхнего и правого нижнего угла bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        objects = ct.update(rects)
    # применение non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    if len(idxs) > 0:

        for objectIDcentroid, rectangle in zip(objects.items(), rects):
            objectID = objectIDcentroid[0]
            text = "ID {}".format(objectID)
            (centerX, centerY, width, height) = rectangle.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] Один фрейм занял {:.4f} секунд".format(elap))
            print("[INFO] Примерное время ожидания завершения обработки: {:.4f} секунд".format(elap * total))

    writer.write(frame)

writer.release()
vs.release()
