# Лабораторная работа №2 — Метрические методы классификации


---

## Цель работы

Изучить и реализовать метрические методы машинного обучения для задачи бинарной классификации на датасете Game of Thrones. Целевая переменная — `isAlive` (жив ли персонаж).

---

## Задачи работы

- **kNN** — метод k ближайших соседей с метрикой Минковского и весами;
- **aNN** — адаптивный метод ближайших соседей (окно Парзена с фиксированной и переменной шириной);
- **ANN** — приближённый поиск ближайших соседей (библиотека Annoy);
- методы улучшения: LOO-подбор гиперпараметров, профиль компактности, отбор эталонных объектов, метод потенциальных функций;
- сравнение с линейными моделями (логистическая регрессия с L2-регуляризацией).

---

## Параметры варианта 17

| Параметр | Значение |
|----------|----------|
| `seed` | 17 |
| `k_max` | 14 |
| `h_max` | 1.4 |
| Метрика kNN | `manhattan` |
| Веса kNN | `uniform` |
| Ядро Parzen | `rectangular` |
| Popularity | `qcut(10)` |
| Culture | объединение редких категорий |
| House | `OneHotEncoder` |
| `n_trees` (Annoy) | 11 |
| `n_neighbors_ann` | 9 |
| `query_character` | Petyr Baelish |

---

## Датасет

- `game_of_thrones_train.csv` — обучающая выборка (1557 строк, 26 столбцов);
- `game_of_thrones_test.csv` — тестовая выборка (389 строк, 25 столбцов, без `isAlive`).

Источник:
- https://raw.githubusercontent.com/EvADarya/Data_processing-DLS-/refs/heads/main/game_of_thrones_train.csv
- https://raw.githubusercontent.com/EvADarya/Data_processing-DLS-/refs/heads/main/game_of_thrones_test.csv

---

## Структура проекта

```
Methods_Algorythms_ML_Lab02/
├── data/                        # CSV-файлы датасета (скачиваются автоматически)
│   ├── game_of_thrones_train.csv
│   └── game_of_thrones_test.csv
├── src/                         # Модули проекта
│   ├── __init__.py
│   ├── config.py                # Параметры варианта 17
│   ├── data_loader.py           # Загрузка данных
│   ├── preprocessing.py         # Предобработка (Часть 1)
│   ├── utils.py                 # Метрики и функции расстояния
│   ├── knn.py                   # kNN (Часть 2)
│   ├── parzen.py                # aNN / окно Парзена (Часть 3)
│   ├── compactness.py           # Профиль компактности (Часть 4)
│   ├── potential.py             # Метод потенциальных функций (Часть 5)
│   ├── ann_search.py            # Annoy (Часть 6)
│   └── logistic.py              # Логистическая регрессия (Часть 7)
├── results/                     # Графики и таблицы результатов
├── reports/                     # Отчёт
├── main.py                      # Главный пайплайн
├── submission.csv               # Предсказания для теста
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/verrronikaat/Methods_Algorythms_ML_Lab02.git
cd Methods_Algorythms_ML_Lab02
```

### 2. Создание виртуального окружения

```bash
python -m venv .venv
```

**Windows:**
```cmd
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Запуск пайплайна

```bash
python main.py
```

Скрипт:
1. Скачает датасеты в `data/`;
2. Выполнит предобработку согласно варианту 17;
3. Обучит и сравнит все модели;
4. Сохранит графики в `results/`;
5. Сформирует `submission.csv` с предсказаниями для тестовой выборки.

---

## Реализованные части

| № | Часть | Модуль |
|---|-------|--------|
| 1 | Подготовка данных | `src/preprocessing.py` |
| 2 | kNN (собственная реализация) | `src/knn.py` |
| 3 | aNN / окно Парзена | `src/parzen.py` |
| 4 | Профиль компактности и отбор эталонов | `src/compactness.py` |
| 5 | Метод потенциальных функций | `src/potential.py` |
| 6 | ANN (Annoy) | `src/ann_search.py` |
| 7 | Логистическая регрессия (L2) | `src/logistic.py` |
| 8 | Формирование `submission.csv` | `main.py` |

---

## Зависимости

См. `requirements.txt`. Основные: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `annoy`, `scipy`.

---

## Воспроизводимость

Все случайные операции зафиксированы через `random_state = 17` (параметр `seed` варианта). Разбиение train/val/test — 70/15/15.

---