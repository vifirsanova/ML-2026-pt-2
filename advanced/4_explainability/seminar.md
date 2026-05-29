# **Интерпретируемость ML: методы атрибуции и объяснения**

**Задача**

Выберите одну статью из списка. Изучите метод и подготовьте доклад продолжительностью 15 минут, включающий анализ работы алгоритма и демонстрацию на данных.

---

**Статьи и методы**

1. Selvaraju et al., 2017 — «Grad-CAM: Visual Explanations from Deep Networks» (Grad-CAM). https://arxiv.org/abs/1610.02391
2. Sundararajan et al., 2017 — «Axiomatic Attribution for Deep Networks» (Integrated Gradients). https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf
3. Tang et al., 2022 — «What the DAAM: Interpreting Stable Diffusion Using Cross Attention» (DAAM). https://arxiv.org/abs/2210.04885
4. Wallace et al., 2019 — AllenNLP Interpret (Gradient Saliency). https://www.researchgate.net/publication/336997924_AllenNLP_Interpret_A_Framework_for_Explaining_Predictions_of_NLP_Models

---

**Рекомендуемые библиотеки**

- Grad-CAM: Captum (PyTorch)
- Integrated Gradients: Captum (PyTorch) или Alibi
- DAAM: Diffusers (HuggingFace)
- AllenNLP Interpret: AllenNLP Interpret

---

**Содержание доклада**

15 минут.

1. Постановка задачи. Какую проблему интерпретируемости решает метод.
2. Объяснение идеи: схема, аналогия, диаграмма, формула с объяснением. Что метод получает на вход и что выдает на выходе.
3. Демонстрация. Ноутбук с работающим кодом. Загрузка модели, запуск метода, визуализация результата. Указание источника реализации.
4. Анализ. Плюсы и минусы метода, подкрепленные конкретными примерами из эксперимента.
5. Возможные улучшения. 
