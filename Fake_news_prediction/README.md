# Определение истинности и ложности новостей по их заголовкам
В нашем распоряжении имееется два файла обучающей и тестовой выборки.

Файл "train.tsv" содержит табличные данные, состояшие из двух колонок: 'title' (заголовки новостей) и 'is_fake' (класс 0 - реальная новость, класс 1 - выдуманная).

В файле test.tsv также есть данные с заголовками новостей, целевую же переменную необходимо предсказать.
# Цель
Построить модель, которая будет классифицировать истинность vs ложность новостей по их заголовкам.
# Библиотеки
pandas, numpy, sklearn, ntlk, re, pymystem3, matplotlib, seaborn, wordcloud