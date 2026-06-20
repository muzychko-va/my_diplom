Музычко Вадим Александрович, КБ-221

# Cистема обнаружения признаков редактирования и монтажа аудиоданных

Приложение для обнаружения монтажа в аудиозаписях. Проект строит mel-спектрограммы из аудио, обучает бинарную CNN-модель и предоставляет простое Tkinter-приложение для проверки отдельных файлов.

## Требования

*   Python 3.9+
*   pip

##  Как запустить

1. Клонируйте репозиторий:
```bash 
git clone https://github.com/muzychko-va/my_diplom.git
```
```bash 
cd diplom
```
2. Установите зависимости:
```bash 
pip install -r requirements.txt
```
3. Запустите проект:
```bash 
python app.py
```

## Дополнительно


* Создать датасет mel-спектрограмм:

```bash
python -m src.build_mel_dataset --dataset-dir Dataset --output-dir mel_dataset
```

* Запустить обучение на подготовленном датасете:

```bash
python -m src.train --manifest mel_dataset/manifest.csv --output-dir models --epochs 20 --batch-size 16
```