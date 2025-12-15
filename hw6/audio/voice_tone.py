import librosa
import numpy as np

def analyze_tone(audio_path: str, text: str = "") -> str:
    y, sr = librosa.load(audio_path, sr=None)

    # Обрезаем тишину
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < sr * 0.25:
        return {
            "tone": "неуверенный",
            "confidence_score": 0.3,
            "feedback": "Команда слишком короткая или тихая"
        }

    # ====== ФИЧИ ======
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)

    # атака (насколько быстро пошла энергия)
    attack_time = np.argmax(rms) / len(rms)

    # ====== НОРМАЛИЗАЦИЯ (смягчённая) ======
    rms_score = np.clip(rms_mean / 0.07, 0, 1)
    zcr_score = np.clip(zcr_mean / 0.12, 0, 1)
    centroid_score = np.clip(centroid_mean / 3500, 0, 1)
    attack_score = 1 - np.clip(attack_time / 0.4, 0, 1)

    confidence = (
        0.4 * rms_score +
        0.2 * zcr_score +
        0.2 * centroid_score +
        0.2 * attack_score
    )

    confidence = float(np.clip(confidence, 0, 1))

    # ====== ИНТЕРПРЕТАЦИЯ ======
    if confidence >= 0.5:
        tone = "уверенный"
        feedback = "Команда дана чётко и уверенно"
    elif confidence >= 0.3:
        tone = "нейтральный"
        feedback = "Команда дана нормально, но можно увереннее"
    else:
        tone = "неуверенный"
        feedback = "В голосе мало уверенности, попробуйте говорить чётче и громче"

    return {
        "tone": tone,
        "confidence_score": round(confidence, 3),
        "feedback": feedback
    }
