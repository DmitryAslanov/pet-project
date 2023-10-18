# импорт библиотек
import cv2
import numpy as np

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights",
                      "dnn_model/yolov4-tiny.cfg")  # инициализация сети

model = cv2.dnn_DetectionModel(net)  # инициализация модели
model.setInputParams(size=(320, 320), scale=1/255)  # параметры модели

# загрузка листов класса
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# инициализация веб-ки
cap = cv2.VideoCapture(0)
# разрешение камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

button_human = False


# функция клика

def click_button(event, x, y, flags, params):
    global button_human
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        polygon = np.array([[(20, 20), (300, 20), (300, 70), (20, 70)]])

        is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
        if is_inside > 0:
            print("Успешное нажатие внутри кнопки", x, y)
            if button_human is False:
                button_human = True
            else:
                button_human = False


# создание окна для клика
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)


# закцикливание кадра для воспроизвидения видео
while True:
    # получение фреймов
    ret, frame = cap.read(0)
    # детекция объекта
    (class_ids, scores, bboxes) = model.detect(frame)
    # цикл для создание рамки объекта и класса
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        # показ классов
        class_name = classes[class_id]

        if class_name == "person" and button_human is True:
            cv2.putText(frame, class_name, (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 2, (150, 50, 50), 2)  # классы
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (200, 0, 50), 3)  # рамка

    # создание интерактивной кнопки
    # cv2.rectangle(frame, (20, 20), (300, 70), (0, 200, 200), -1)
    polygon = np.array([[(20, 20), (300, 20), (300, 70), (20, 70)]])
    cv2.fillPoly(frame, polygon, (0, 0, 200))
    cv2.putText(frame, "human", (30, 60),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255), 3)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
