from ultralytics import YOLO
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (102, 217, 255)

"""отрисовка парковочных зон"""
def parking_zone1(frame, parking_car):
    if parking_car >= 7:
        cv2.line(frame, (60, 85), (140, 65), RED, thickness=2)
        cv2.line(frame, (140, 65), (260, 260), RED, thickness=2)
        cv2.line(frame, (260, 260), (150, 285), RED, thickness=2)
        cv2.line(frame, (60, 85), (150, 285), RED, thickness=2)
    elif parking_car == 6:
        cv2.line(frame, (60, 85), (140, 65), YELLOW, thickness=2)
        cv2.line(frame, (140, 65), (260, 260), YELLOW, thickness=2)
        cv2.line(frame, (260, 260), (150, 285), YELLOW, thickness=2)
        cv2.line(frame, (60, 85), (150, 285), YELLOW, thickness=2)
    else:
        cv2.line(frame, (60, 85), (140, 65), GREEN, thickness=2)
        cv2.line(frame, (140, 65), (260, 260), GREEN, thickness=2)
        cv2.line(frame, (260, 260), (150, 285), GREEN, thickness=2)
        cv2.line(frame, (60, 85), (150, 285), GREEN, thickness=2)
        cv2.putText(frame, str(7 - parking_car), (210, 170), font, 2, GREEN, 1)
    return frame


def parking_zone2(frame, parking_car):
    if parking_car == 1:
        cv2.line(frame, (20, 310), (90, 310), RED, thickness=2)
        cv2.line(frame, (90, 310), (70, 400), RED, thickness=2)
        cv2.line(frame, (70, 400), (0, 400), RED, thickness=2)
        cv2.line(frame, (0, 400), (20, 310), RED, thickness=2)
    else:
        cv2.line(frame, (20, 310), (90, 310), YELLOW, thickness=2)
        cv2.line(frame, (90, 310), (70, 400), YELLOW, thickness=2)
        cv2.line(frame, (70, 400), (0, 400), YELLOW, thickness=2)
        cv2.line(frame, (0, 400), (20, 310), YELLOW, thickness=2)
    return frame


def parking_zone3(frame, parking_car):
    if parking_car >= 4:
        cv2.line(frame, (2, 500), (478, 500), RED, thickness=2)
        cv2.line(frame, (140, 65), (260, 260), RED, thickness=2)
        cv2.line(frame, (260, 260), (150, 285), RED, thickness=2)
        cv2.line(frame, (60, 85), (150, 285), RED, thickness=2)
    elif parking_car == 3:
        cv2.line(frame, (2, 500), (600, 500), YELLOW, thickness=2)
        cv2.line(frame, (2, 500), (2, 570), YELLOW, thickness=2)
        cv2.line(frame, (2, 570), (600, 570), YELLOW, thickness=2)
        cv2.line(frame, (598, 500), (598, 570), YELLOW, thickness=2)
        cv2.putText(frame, str(4 - parking_car), (280, 630), font, 2, YELLOW, 1)
    else:
        cv2.line(frame, (2, 500), (600, 500), GREEN, thickness=2)
        cv2.line(frame, (2, 500), (2, 600), GREEN, thickness=2)
        cv2.line(frame, (2, 570), (600, 570), GREEN, thickness=2)
        cv2.line(frame, (600, 500), (600, 570), GREEN, thickness=2)
        cv2.putText(frame, str(4 - parking_car), (530, 300), font, 2, GREEN, 1)
    return frame


model = YOLO("./models/yolov8m.pt")

# cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.1:554/media/video1', cv2.CAP_FFMPEG)

while True:
    park_1 = 0
    park_2 = 0
    park_3 = 0
    frame = cv2.imread("./images/cars.jpeg")
    frame = cv2.resize(frame, (600, 800))
    results = model.predict(frame)
    result = results[0]
    for box in result.boxes:
        conf = box.conf[0].item()
        conf = round(conf * 100)
        class_id = result.names[box.cls[0].item()]

        if conf > 35:
            for x, y, w, h in box.xyxy.tolist():
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                """Счет машин на P1"""
                if (60 < x < 150) and (70 < y < 280):
                    park_1 += 1
                """Счет машин на P2"""
                if (0 < x < 90) and (300 < y < 360):
                    park_2 += 1
                """Счет машин на P3"""
                if (0 <= x < 600) and (490 < y < 550):
                    park_3 += 1

                frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), thickness=1)
                font = cv2.FONT_HERSHEY_DUPLEX

    parking_zone1(frame, park_1)
    parking_zone2(frame, park_2)
    parking_zone3(frame, park_3)

    cv2.imshow("video feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("./images/parking_finish.jpeg", frame)
        break

# cap.release()
cv2.destroyAllWindows()
