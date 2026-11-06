# SportsClassification

Классификация фото на 100 видов спорта на PyTorch: обучение в Google Colab, предсказание через FastAPI, интерфейс на Streamlit.

## Набор данных

100 Sports Image Classification (`archive.zip`, он же `gpiosenka/sports-classification` на Kaggle):
train 13 492 фото, valid 500, test 500, по 5 фото на класс в test.

## Обучение

`sports_classification.ipynb` - ноутбук для Colab (GPU T4), сделан по шаблону урока:

- `ImageFolder`, фото 128×128; в train случайная вырезка `RandomResizedCrop(scale=(0.7, 1.0))`, отражение и поворот до 10°;
- своя свёрточная сеть `CheckSportsModel`: `first` - 8 слоёв Conv + BatchNorm + ReLU с MaxPool, `second` - Dropout и два Linear до 100 классов;
- `CrossEntropyLoss(label_smoothing=0.1)`, Adam lr 0.001 с `CosineAnnealingLR`, 40 эпох (~15 мин на T4);
- точность на test: **83,4%**, правильный класс в топ-3 - 94,6%.

Запуск в Colab: загрузить ноутбук, выбрать T4 GPU, «Выполнить все». Первая ячейка сама скачивает `archive.zip` с Kaggle (без логина).
В конце сохраняются `sports_model.pth` (веса) и `sports_classes.pth` (названия классов).

## Запуск сервиса

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
.venv/bin/streamlit run front.py
```

`sports_model.pth` и `sports_classes.pth` должны лежать в корне проекта.
FastAPI: http://127.0.0.1:8000/docs, Streamlit: http://localhost:8501.

## API

`POST /sports_predict/` - фото в поле `file` (multipart/form-data).

```json
{
  "class_id": 31,
  "label": "football",
  "confidence": 54.41,
  "top3": [
    {"class_id": 31, "label": "football", "confidence": 54.41},
    {"class_id": 53, "label": "lacrosse", "confidence": 16.68},
    {"class_id": 7, "label": "baseball", "confidence": 7.17}
  ]
}
```

Если файл не картинка - `400 Файл не является изображением`.
