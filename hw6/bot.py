import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types

from audio.whisper_asr import audio_to_text
from audio.voice_tone import analyze_tone

from video.pose_detector import evaluate_pose_from_image

from ultralytics import YOLO
from bot_info.config import BOT_TOKEN

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# временное хранение состояния
USER_LAST_COMMAND = {}


# /start
@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    text = (
        "Привет! \n"
        "Я DogTrainer AI — умный помощник для дрессировки собак \n\n"

        "Что я умею:\n"
        "Анализировать голосовые команды:\n"
        "  • какая команда была дана\n"
        "  • насколько уверенно и чётко она произнесена\n\n"

        "Анализировать фото/видео:\n"
        "  • есть ли собака в кадре\n"
        "  • есть ли человек в кадре\n"
        "  • какую позу выполняет собака\n"
        "  • верно ли выполнена команда\n\n"

        "В ответ ты получишь:\n"
        "  • оценку\n"
        "  • комментарий и подсказки\n\n"

        "Важно:\n"
        "Обработка фото и аудио может занять до 10–20 секунд.\n\n"

        "Как пользоваться:\n"
        "  1. Отправь голосовую команду\n"
        "  2. Затем отправь фото или видео выполнения команды\n\n"

        "Готов начинать! Поехали!"
    )

    await message.answer(text)

# Голос
@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice(message: types.Message):
    await message.answer("Анализирую голос, подожди немного…")

    voice = message.voice
    file = await bot.get_file(voice.file_id)
    file_path = file.file_path

    local_audio_path = f"temp/{message.from_user.id}.ogg"
    await bot.download_file(file_path, local_audio_path)

    text = audio_to_text(local_audio_path)
    tone = analyze_tone(local_audio_path, text)


    if text == " Сидеть" or text == " сидеть" or text == " сидеть!" or text == " сидеть." or text == " сидеть,":
        USER_LAST_COMMAND[message.from_user.id] = "sit"
    elif text == " Лежать" or text == " лежать" or text == " лежать!" or text == " лежать." or text == " лежать,":
        USER_LAST_COMMAND[message.from_user.id] = "lie"
    elif text == " Стоять" or text == " стоять" or text == " стоять!" or text == " стоять." or text == " стоять,":
        USER_LAST_COMMAND[message.from_user.id] = "stay"
    else:
        USER_LAST_COMMAND[message.from_user.id] = "unknown"

    reply = (
        f"Распознанная команда:\n"
        f"«{text}»\n\n"
        f"Оценка тона:\n"
        f"  - Уверенность: {tone['tone']}\n"
        f"  - Балл уверенности: {tone['confidence_score']}\n"
        f"  - Рекомендации: {tone['feedback']}\n\n"
        f"Теперь отправь фото или видео выполнения команды"
    )

    await message.answer(reply)

# Фото
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    await message.answer("Анализирую фото, пожалуйста подожди…")

    user_id = message.from_user.id

    if user_id not in USER_LAST_COMMAND:
        await message.answer(
            "Сначала отправь голосовую команду, а потом фото"
        )
        return

    expected_command = USER_LAST_COMMAND[user_id]

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    image_path = f"temp/{user_id}.jpg"
    await bot.download_file(file.file_path, image_path)

    result = evaluate_pose_from_image(
        image_path=image_path,
        expected_command=expected_command
    )

    # Ответ
    if result.get('detected_pose') == None:
        reply = (
            f"Результат анализа видео:\n\n"
            f"Поза: Не распознана\n"
            f"Оценка: Ошибка\n"
            f"Баллы: {result.get('score')}\n\n"
            f"Комментарий:\n{result.get('feedback')}"
        )
    else:
        reply = (
            f"Результат анализа видео:\n\n"
            f"Поза: {result.get('detected_pose')}\n"
            f"Уверенность: {result.get('confidence')}\n"
            f"Оценка: {result.get('grade')}\n"
            f"Баллы: {result.get('score')}\n\n"
            f"Комментарий:\n{result.get('feedback')}"
        )

    await message.answer(reply)

    # Отправляем рекомендации отдельно
    if result.get('grade') == "Ошибка" or result.get('detected_pose') == None:
        recommendation_text = (
            f"Команда выполнена неверно! \n\n"
            f"Возможные причины:\n"
            f" - не знание техники выполнения команды;\n"
            f" - уставшая/невыгуленная собака;\n"
            f" - не четко озвучена команда;\n"
            f" - отвлекающие элементы.\n"
            f"Попрактикуйся и отправь результат тренировки еще раз.\n\n У тебя все получится!"
        )
    elif result.get('grade') == "Хорошо" or result.get('grade') == "Удовлетворительно":
        recommendation_text = (
            f"Ты молодец! Рекомендую больше тренироваться, так как все еше есть неточности. Возможно стоит попробовать другие техники)"
        )
    elif result.get('grade') == "Отлично":
        recommendation_text = (
            f"Ты умница! Так держать! Не забывай тренироваться каждый день идя домой с прогулки"
        )

    await message.answer(recommendation_text)

# Видео
@dp.message_handler(content_types=types.ContentType.VIDEO)
async def handle_video(message: types.Message):
    await message.answer("Анализирую видео, пожалуйста, подожди…")

    user_id = message.from_user.id

    if user_id not in USER_LAST_COMMAND:
        await message.answer(
            "Сначала отправь голосовую команду, а потом видео"
        )
        return

    expected_command = USER_LAST_COMMAND[user_id]

    video = message.video
    file = await bot.get_file(video.file_id)

    video_path = f"temp/{user_id}.mp4"
    await bot.download_file(file.file_path, video_path)

    # Здесь вызываем функцию анализа видео (аналог evaluate_pose_from_image)
    result = evaluate_pose_from_image(
        image_path=video_path,
        expected_command=expected_command
    )

    if result.get('detected_pose') == None:
        reply = (
            f"Результат анализа видео:\n\n"
            f"Поза: Не распознана\n"
            f"Оценка: Ошибка\n"
            f"Баллы: {result.get('score')}\n\n"
            f"Комментарий:\n{result.get('feedback')}"
        )
    else:
        reply = (
            f"Результат анализа видео:\n\n"
            f"Поза: {result.get('detected_pose')}\n"
            f"Уверенность: {result.get('confidence')}\n"
            f"Оценка: {result.get('grade')}\n"
            f"Баллы: {result.get('score')}\n\n"
            f"Комментарий:\n{result.get('feedback')}"
        )

    await message.answer(reply)

    # Отправляем рекомендации отдельн
    if result.get('grade') == "Ошибка" or result.get('detected_pose') == None:
        recommendation_text = (
            f"Команда выполнена неверно! \n\n"
            f"Возможные причины:\n"
            f" - не знание техники выполнения команды;\n"
            f" - уставшая/невыгуленная собака;\n"
            f" - не четко озвучена команда;\n"
            f" - отвлекающие элементы.\n"
            f"Попрактикуйся и отправь результат тренировки еще раз.\n\n У тебя все получится!"
        )
    elif result.get('grade') == "Хорошо" or result.get('grade') == "Удовлетворительно":
        recommendation_text = (
            f"Ты молодец! Рекомендую больше тренироваться, так как все еше есть неточности. Возможно стоит попробовать другие техники)"
        )
    elif result.get('grade') == "Отлично":
        recommendation_text = (
            f"Ты умница! Так держать! Не забывай тренироваться каждый день идя домой с прогулки"
        )

    await message.answer(recommendation_text)

# Запуск
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
