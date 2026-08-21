"""Video üzerinde telefonla konuşan kişileri tespit eder."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR.parent / "videos" / "video2.mp4"
MODELS_DIR = SCRIPT_DIR.parent / "models"

PERSON_CONF_THRESHOLD = 0.4
PHONE_CONF_THRESHOLD = 0.1
CELL_PHONE_LABEL = "cell phone"

# COCO poz iskeletindeki omuz nokta indeksleri.
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_TEXT = "Telefonla Konusuluyor"

phone_model = YOLO(str(MODELS_DIR / "yolo26l.pt"))
pose_model = YOLO(str(MODELS_DIR / "yolo26n-pose.pt"))


def telefontespiti(img):
    """Görüntüdeki telefonların bbox koordinatlarını döndürür."""
    bbox_list = []

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = phone_model(rgb_img, verbose=False)
    labels = results[0].names

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        label = results[0].boxes.cls[i]
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)

        if score < PHONE_CONF_THRESHOLD:
            continue
        if labels[label] == CELL_PHONE_LABEL:
            bbox_list.append((x1, y1, x2, y2))

    return bbox_list


def insantespiti(img):
    """PERSON_CONF_THRESHOLD üzerindeki her kişi için (tam bbox, omuz hizasına
    daraltılmış bölge, poz noktaları, güven skoru) dörtlüsünü döndürür."""
    box_list = []
    score_list = []
    region_list = []
    keypoint_list = []
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model(rgb_img, verbose=False)

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_list.append((x1, y1, x2, y2))
        score_list.append(float(results[0].boxes.conf[i]))

        region = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        region = region.reshape((-1, 1, 2))
        region_list.append(region)

    # Bölgenin alt sınırını, kişinin bounding box'ından omuz hizasına çekiyoruz.
    for j, keypoints in enumerate(results[0].keypoints.xy):
        noktalar = []
        for i, keypoint in enumerate(keypoints):
            if i not in (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX):
                continue
            x3, y3 = keypoint
            x3, y3 = int(x3), int(y3)
            if x3 == 0 and y3 == 0:
                continue
            noktalar.append((x3, y3))
            if i == LEFT_SHOULDER_IDX:
                region_list[j][2][0][1] = y3
            if i == RIGHT_SHOULDER_IDX:
                region_list[j][3][0][1] = y3
        keypoint_list.append(noktalar)

    kisiler = zip(box_list, region_list, keypoint_list, score_list)
    return [kisi for kisi in kisiler if kisi[3] >= PERSON_CONF_THRESHOLD]


def etiket_ciz(img, text):
    """Sol üst köşeye dolgulu arka planlı, beyaz yazılı etiket çizer."""
    (w, h), baseline = cv2.getTextSize(text, FONT, 0.8, 2)
    x, y = 10, 10
    cv2.rectangle(img, (x, y), (x + w + 16, y + h + baseline + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 8, y + h + 8), FONT, 0.8, (255, 255, 255), 2)


def main():
    kamera = cv2.VideoCapture(str(VIDEO_PATH))

    try:
        while True:
            ret, kare = kamera.read()
            if not ret:
                break

            kisi_listesi = insantespiti(kare)

            konusuluyor = False

            for (kutu, bolge, noktalar, _) in kisi_listesi:
                bx1, by1, bx2, by2 = kutu
                cv2.rectangle(kare, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                for (px, py) in noktalar:
                    cv2.circle(kare, (px, py), 3, (0, 255, 255), -1)

            if kisi_listesi:
                # Telefon tespiti, en yüksek güven skorlu kişinin bbox'ıyla kırpılmış
                # görüntü üzerinde yapılır; kırpılmış görüntü kareler arasında saklanmaz.
                kutu, bolge, _, _ = max(kisi_listesi, key=lambda kisi: kisi[3])
                x1, y1, x2, y2 = kutu
                h, w = kare.shape[:2]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)

                if x2 > x1 and y2 > y1:
                    kirpilmis = kare[y1:y2, x1:x2]

                    for (px1, py1, px2, py2) in telefontespiti(kirpilmis):
                        px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
                        cv2.rectangle(kare, (px1, py1), (px2, py2), (255, 0, 0), 2)

                        cx, cy = int(px1 / 2 + px2 / 2), int(py1 / 2 + py2 / 2)
                        if cv2.pointPolygonTest(bolge, (cx, cy), False) > 0:
                            konusuluyor = True
                            cv2.polylines(kare, [bolge], True, (102, 0, 153), 3)

            if konusuluyor:
                etiket_ciz(kare, LABEL_TEXT)

            cv2.imshow("kamera", kare)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
