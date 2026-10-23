# Домашнее задание: самостоятельное освоение Hugging Face Transformers

## Цель

Разобрать официальные туториалы Hugging Face по тонкой настройке трансформеров, выбрать тему классификации текста, обучить модель и подготовить отчёт. Мультимодальные системы разбираются отдельно на следующем занятии.

---

## Часть 1. Самостоятельный разбор туториалов Hugging Face

### 1.1. Обязательные материалы

Изучите следующие страницы официальной документации и курсов Hugging Face:

| № | Материал | Что извлечь |
|---|---|---|
| 1 | [Text classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) | Полный пайплайн: загрузка данных, токенизация, `Trainer`, `compute_metrics` |
| 2 | [Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer) | Параметры `TrainingArguments`, `compute_metrics`, `EarlyStoppingCallback` |
| 3 | [Datasets: load local CSV/JSON](https://huggingface.co/docs/datasets/loading) | `load_dataset("csv", data_files={...})` для собственных данных |
| 4 | [Evaluate: F1 metric](https://huggingface.co/docs/evaluate) | Разница `micro` / `macro` / `weighted` усреднения |

### 1.2. Задание на разбор

Для каждого материала выпишите в отчёт (кратко, 3–5 предложений):

1. **Что делает** данный компонент/функция.
2. **Какие параметры** критичны для классификации текста и почему.
3. **Один пример кода** из туториала, который вы планируете использовать.

**Проверка понимания.** Ответьте на вопросы:

- Чем `problem_type="single_label_classification"` отличается от `"multi_label_classification"` с точки зрения функции потерь? 
- Зачем нужен `processing_class=tok` в `Trainer` (а не `tokenizer=tok`)? 
- Что произойдёт, если не передать `compute_metrics` в `Trainer`?

---

## Часть 2. Выбор темы и обучение классификатора

### 2.1. Варианты тем (выберите одну или предложите свою)

| № | Тема | Датасет (Hugging Face) | Число классов | Особенность |
|---|---|---|---|---|
| 1 | Классификация новостей | `ag_news` | 4 | Многоклассовая, сбалансированная |
| 2 | Анализ тональности отзывов | `imdb` или `sst2` | 2 | Бинарная, длинные тексты |
| 3 | Определение спама | `sms_spam` | 2 | Короткие тексты, дисбаланс |
| 4 | Классификация научных статей | `arxiv` (подвыборка категорий) | 3–5 | Длинные тексты, техническая лексика |
| 5 | Эмоции в твитах | `go_emotions` (single-label вариант) | 27 → выберите 5–7 | Многоклассовая, редкие классы |
| 6 | Своя тема | Любой датасет с полем `text` и `label` | ≥ 2 | Согласовать с преподавателем |

### 2.2. Требования к обучению

1. **Модель:** `distilbert-base-uncased` или `bert-base-uncased` (для экономии времени — DistilBERT).
2. **Данные:** загрузите датасет через `load_dataset`, разделите на train/validation (если в датасете нет готового validation split).
3. **Токенизация:** `truncation=True`, `max_length=128` (или 256 для длинных текстов).
4. **Метрики:** `accuracy` и `f1_macro`. Для многоклассовой задачи `f1_macro` важнее accuracy при дисбалансе.
5. **Early stopping:** `EarlyStoppingCallback(early_stopping_patience=2)` с `metric_for_best_model="f1_macro"` .
6. **Гиперпараметры:** `learning_rate=2e-5`, `num_train_epochs=3–5`, `per_device_train_batch_size=16`.

### 2.3. Что сдать по части 2

- Код обучения (один блок в ноутбуке или `.py`-файл).
- Таблица результатов: `eval_accuracy`, `eval_f1_macro`, `eval_loss` для финальной модели.
- Confusion matrix (для многоклассовых задач).

**Задание на подумать.** Сравните `f1_macro` и `accuracy`. Если `f1_macro` заметно ниже `accuracy`, какие классы модель путает? Выпишите 2–3 пары confused classes.

---

## Часть 3. Анализ ошибок (краткий)

Для 5 случайно выбранных ошибочных примеров из validation set выпишите:

| Текст (первые 100 символов) | Истинный класс | Предсказанный класс | Гипотеза о причине |
|---|---|---|---|
| ... | ... | ... | ... |

**Гипотезы формулируйте конкретно:** «короткий текст, мало контекста», «смешение тематик Business / Sci-Tech», «редкая лексика вне словаря», «класс minority, модель смещена к majority».

---

## Часть 4. Подготовка к следующему занятию (мультимодальные системы)

Перед следующим семинаром прочитайте:

1. **Hugging Face blog: Vision Language Models (VLMs) 2025** — обзор архитектур VLM, ключевые компоненты (vision encoder + LLM) .
2. **Transformers docs: Multimodal chat templates** — как `Processor` обрабатывает изображения и текст совместно .
3. **DataCamp course outline: Multi-Modal Models with Hugging Face** — раздел «Multi-Modal Models for Classification» (CLIP, zero-shot) .

**Вопрос для размышления.** В текстовой классификации вы передаёте в модель `input_ids` и `attention_mask`. В мультимодальной модели с изображением и текстом — что добавляется? Как вы думаете, почему `Processor` (а не `Tokenizer`) отвечает за подготовку входов? 

---

## Критерии оценивания

| Критерий | Вес |
|---|---|
| Часть 1: разбор туториалов, ответы на вопросы | 20% |
| Часть 2: обучение модели, метрики, confusion matrix | 40% |
| Часть 3: анализ 5 ошибок с конкретными гипотезами | 20% |
| Часть 4: подготовка к мультимодальной теме | 20% |

**Штрафы:**
- Модель не обучена / код не запускается — до 50% от веса части 2.
- Гипотезы об ошибках общие («модель не поняла текст») без привязки к классам и данным — до 50% от веса части 3.
- Отсутствие `f1_macro` в метриках — до 25% от веса части 2.

---

## Полезные ссылки

- [Hugging Face Transformers: Text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) 
- [Hugging Face Datasets: Loading](https://huggingface.co/docs/datasets/loading)
- [Hugging Face Evaluate: F1](https://huggingface.co/docs/evaluate)
- [Hugging Face LLM Course (Chapter 1: Transformers)](https://huggingface.co/learn/llm-course) 
- [Multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal) 
- [Vision Language Models 2025](https://huggingface.co/blog/vlms-2025) 
