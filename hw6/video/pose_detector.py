from ultralytics import YOLO

dog_model = YOLO("models/best.pt")         
person_model = YOLO("models/yolo11m.pt")    

import numpy as np

COMMAND_TO_POSE = {
    "sit": "sit",
    "lie": "lie",
    "stay": "stay"
}

def evaluate_pose_from_image(
    image_path: str,
    expected_command: str,
    dog_model=dog_model,
    person_model=person_model,
    conf_threshold = 0.6
):
    result = {
        "dog_detected": False,
        "person_detected": False,
        "detected_pose": None,
        "confidence": None,
        "command_correct": False,
        "score": 0,
        "grade": None,
        "feedback": None
    }

    # 1. Детекция человека
    person_res = person_model(image_path)[0]
    if person_res.boxes is not None:
        for cls, conf in zip(person_res.boxes.cls, person_res.boxes.conf):
            if int(cls) == 0 and conf > 0.5:  # class 0 = person
                result["person_detected"] = True
                break

    # 2. Детекция собаки
    dog_res = dog_model(image_path)[0]

    if dog_res.boxes is None or len(dog_res.boxes) == 0:
        result["feedback"] = "Поза не распознана"
        return result

    # берём самую уверенную детекцию
    best_idx = np.argmax(dog_res.boxes.conf.cpu().numpy())
    box = dog_res.boxes[best_idx]

    detected_pose = dog_model.names[int(box.cls)]
    confidence = float(box.conf)

    result["dog_detected"] = True
    result["detected_pose"] = detected_pose
    result["confidence"] = round(confidence, 3)

    if confidence < conf_threshold:
        result["feedback"] = "Неуверенное распознавание позы"
        return result

    # 3. Проверка команды
    expected_pose = COMMAND_TO_POSE.get(expected_command)

    if detected_pose != expected_pose:
        result["feedback"] = "Команда выполнена неверно"
        result["grade"] = "Ошибка"
        return result

    result["command_correct"] = True

    # 4. Геометрическая оценка
    x, y, w, h = box.xywh[0].cpu().numpy()
    aspect_ratio = h / w

    geometry_score = 0

    if detected_pose == "sit" and 1.1 <= aspect_ratio <= 1.6:
        geometry_score = 20
    elif detected_pose == "lie" and aspect_ratio > 1.6:
        geometry_score = 20
    elif detected_pose == "stay" and 0.8 <= aspect_ratio <= 1.2:
        geometry_score = 20
    else:
        geometry_score = 10

    # 5. Финальный скоринг
    score = 50                     # за правильную позу
    score += min(confidence * 30, 30)
    score += geometry_score

    result["score"] = int(score)

    if score >= 85:
        result["grade"] = "Отлично"
        result["feedback"] = "Команда выполнена идеально"
    elif score >= 65:
        result["grade"] = "Хорошо"
        result["feedback"] = "Хорошо, но можно выполнить ровнее"
    else:
        result["grade"] = "Удовлетворительно"
        result["feedback"] = "Почти получилось, попробуйте ещё раз"

    # 6. Подсказка про человека
    if not result["person_detected"]:
        result["feedback"] += " (хозяин не обнаружен в кадре)"

    return result
