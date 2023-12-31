# ML_HW_1
Домашка №1 по ML, посвящённая линейной регрессии и FastAPI

### Что содержит репозиторий?
- HW1_Regression_with_inference.ipynb - исходник домашки с ML частью
- HW1_Regression_with_inference.html - HTML версия домашки с ML частью
- main.py - исходник FastAPI сервиса 
- Файлы моедли 
  - Ridge_model.pkl - веса модели
  - Ridge_scaler.pkl - скейлин факторов
  - Ridge_encoder.pkl - OneHotEncoding факторов
- Исходники датасета
  - cars_test.csv
  - cars_test.csv

## Выводы о проделанной работе:
#### что было сделано
1. Базовый EDA и предобработка признаков
2. Визуализация зависимостей признаков и таргета 
3. Обучение Линейной Регрессии в следующем виде:
  - базовый LinearRegression только на вещественных признаках, без нормировки данных
  - базовый LinearRegression только на вещественных признаках, с нормировкой данных
  - Lasso линейная регерссия только на вещественных признаках, с нормировкой данных, с ипользованием перебора по сетке
  - ElasticNet линейная регерссия только на вещественных признаках, с нормировкой данных, с ипользованием перебора по сетке
  - Ridge линейная регерссия c OneHotEncoding категорриальных признаков и нормировкой данных, с ипользованием перебора по сетке
4. Написание бизнесовой метрики качества (отклонение предсказания от факта не более чем на 10%)
5. Реализация сервиса на FastAPI

#### с какими результатами
В ходе обучения линейной регрессии было получено среднее качество предсказаний. Это качество определённо можно улучшить за счёт Feature Engineering, удаления выбросов, а также избавления от линейно зависимости фичей max_power и engine 🙃
Наилучший $R^2$, который удалось выбить на тесте это ~ 0.506
ElasticNet выдал худшую метрику ушедшую в минус 
Все остальные значения были на уровне ~ 0.41

#### что дало наибольший буст в качестве
Самый большой буст в качесве дало добавление категориальных фичей и использование Ridge регресии 

#### что сделать не вышло и почему (это нормально, даже хорошо😀)
Не удалось позаниматься Feature Engineering, тк на него просто не нашлось моральных сил, хотя это довольно творческий процесс. Который в другой ситуации мог бы вызвать у меня большой интерес

## FastAPI
При написание FastAPI предполагается, что json объект в себе может сожержать значение цены равное 0, но этот объект удаляется внутри методов. В целом можно было удалить selling_price из pydantic объекта, но поле было оставлено для удобства тестирование.  

Демонстрация работы метода @app.post("/predict_item") 
![image](https://github.com/BedPled/ML_HW_1/assets/65103970/fdffd029-88bb-4a6c-842a-b0f08258552e)

Демонстрация работы метода @app.post("/predict_items")
![image](https://github.com/BedPled/ML_HW_1/assets/65103970/a8a845eb-6f12-4358-8294-f4c24289cd37)
