import json
import random
import logging
import re
import torch
from aiogram import Bot, Dispatcher, executor, types

from model.model import NeuralNet
from model.nltk_utils import tokenize, bag_of_words

# --- Логирование ---
logging.basicConfig(filename="logs/telegram_chat.log", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Телеграм ---
TOKEN = "8222404669:AAHnSWnuK94pjxHnsRbTYAbWEQEVeN48Lg0"
bot = Bot(TOKEN)
dp = Dispatcher(bot)

# --- Память пользователей ---
user_dogs = {}

# ------------------------------------------------------
#   Расчёт нормы питания
# ------------------------------------------------------
def calculate_food(text):
    text = text.lower()

    # --- Извлекаем вес ---
    weight = None
    w = re.search(r'(\d+(\.\d+)?)\s*кг', text)
    if w:
        weight = float(w.group(1))

    # --- Извлекаем возраст ---
    age = None
    age_match = re.search(r'(\d+)\s*(год|года|лет|месяц|месяца|месяцев)', text)
    if age_match:
        val = int(age_match.group(1))
        unit = age_match.group(2)
        age = val / 12 if "месяц" in unit else val

    # --- Тип питания ---
    if "натурал" in text:
        diet = "natural"
    elif "комбо" in text:
        diet = "combo"
    else:
        diet = "dry"

    # --- Кастрация ---
    neutered = any(x in text for x in ["кастр", "стерил"])

    if not weight or not age:
        return "Укажи, пожалуйста, вес собаки (например: 12 кг), возраст (например: 8 месяцев или 3 года), была ли стерилизация/кастрация и если была назначена диета - указать (натуралка/сухой/комбо)"

    # --- РАСЧЁТ ---
    RER = 70 * (weight ** 0.75)
    MER = RER * 3 if age < 1 else RER * 1.6

    if neutered:
        MER *= 0.85

    dry = round(MER / 3)
    natural = round(MER / 1.5)
    combo_dry = round(dry * 0.6)
    combo_nat = round(natural * 0.4)

    return (
        f"**Расчёт нормы питания**\n"
        f"Вес: {weight} кг, возраст: {age:.1f} года\n"
        f"Тип питания: {'натуралка' if diet=='natural' else 'сухой' if diet=='dry' else 'комбо'}\n"
        f"{'Стерилизована/кастрирована — порции уменьшены.' if neutered else ''}\n\n"
        f"Суточная калорийность: **{int(MER)} ккал**\n\n"
        f"Натуралка: **{natural} г/сут**\n"
        f"Сухой корм: **{dry} г/сут**\n"
        f"Комбо: **{combo_dry} г сухого + {combo_nat} г натуралки**"
    )

# ------------------------------------------------------
#   Модель
# ------------------------------------------------------
device = torch.device('cpu')
data = torch.load("model/data.pth", map_location=device)

model = NeuralNet(
    data["input_size"], 
    data["hidden_size"], 
    data["output_size"]
).to(device)

model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)


def get_response(msg):
    msg_low = msg.lower().strip()

    # Еда
    if ("кг" in msg_low) or ("корм" in msg_low):
        return calculate_food(msg_low)

    # Повтор
    if any(word in msg_low for word in ["снова", "повтори", "еще раз", "ещё раз"]):
        return None  # обработается в хэндлере

    # Модель
    sentence = tokenize(msg_low)
    X = torch.tensor(bag_of_words(sentence, all_words), dtype=torch.float).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)

    if probs[0][predicted.item()] > 0.6:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    return "Извини, я пока не понял тебя"


# ------------------------------------------------------
#   Меню /start
# ------------------------------------------------------
@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add("Дрессировка", "Поведение")
    keyboard.add("Опасность для здоровья")

    await message.answer(
        "Привет! Я *Dog Trainer Assistant*\n"
        "Помогу с дрессировкой, поведением и здоровьем.\n\n"
        "Выбери раздел:",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


# ------------------------------------------------------
#   Меню категорий
# ------------------------------------------------------

@dp.message_handler(text="Дрессировка")
async def training_menu(message: types.Message):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("Сидеть", "Лежать", "Стоять", "Рядом")
    kb.add("Фу", "Апорт", "Барьер", "Гуляй")
    kb.add("В меню")
    await message.answer("Раздел *Дрессировка*", reply_markup=kb, parse_mode="Markdown")


@dp.message_handler(text="Поведение")
async def behavior_menu(message: types.Message):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("Дрожит от страха", "Встряхивается как после купания")
    kb.add("Кусается", "Виляет хвостом")
    kb.add("В меню")
    await message.answer("Раздел *Поведение*", reply_markup=kb, parse_mode="Markdown")


@dp.message_handler(text="Опасность для здоровья")
async def health_menu(message: types.Message):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("Собака не дышит", "Собака потеряла сознание")
    kb.add("Массаж сердца собаке", "Искусственное дыхание собаке")
    kb.add("В меню")
    await message.answer("Раздел *Опасность для здоровья*", reply_markup=kb, parse_mode="Markdown")


@dp.message_handler(text="В меню")
async def back_to_menu(message: types.Message):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("Дрессировка", "Поведение")
    kb.add("Опасность для здоровья")
    await message.answer("Выбери раздел:", reply_markup=kb)


# ------------------------------------------------------
#   Основной обработчик сообщений
# ------------------------------------------------------
@dp.message_handler()
async def main_handler(message: types.Message):
    uid = message.from_user.id
    msg = message.text

    # Предварительный ответ модели или расчёта
    reply = get_response(msg)

    # Если был "повтори"
    if reply is None:
        if uid in user_dogs:
            await message.answer(calculate_food(user_dogs[uid]), parse_mode="Markdown")
        else:
            await message.answer("Мне нечего повторять")
        return

    # Сохраняем запрос с весом
    if "кг" in msg or "корм" in msg:
        user_dogs[uid] = msg

    # Отправляем ответ
    await message.answer(reply, parse_mode="Markdown")

    # Логи
    logging.info(f"USER({uid}): {msg}")
    logging.info(f"BOT: {reply}")


# ------------------------------------------------------
#   Запуск
# ------------------------------------------------------
if __name__ == "__main__":
    print("Бот запущен...")
    executor.start_polling(dp, skip_updates=True)
