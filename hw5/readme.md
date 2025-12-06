# Dog → Artistic Style Transfer (Van Gogh, Monet)
Локальный проект CycleGAN под macOS для переноса художественных стилей на фотографии собаки. Используются два стиля:

1. Van Gogh
2. Monet

## Установка
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Подготовка данных
Положите фото собаки в:
data/dogs_raw/

Запустите:
python scripts/prepare_data.py

Скрипт:
- сделает аугментации
- ресайз 256×256
- положит результат в data/dogs_preprocessed/
- автоматически распределит в trainA/testA

## Обучение моделей
Van Gogh:
python scripts/train_vangogh.py

Monet:
python scripts/train_monet.py

## Тестирование / генерация
Van Gogh:
python scripts/test_vangogh.py

Monet:
python scripts/test_monet.py

## Результаты сохраняются в:
results/vangogh/
results/monet/
