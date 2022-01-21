# Проект - детекция игральных карт

Данный проект выполнен в рамках обучения в Deep Learning School от МФТИ. 

Ссылка на развернутое приложение:
https://share.streamlit.io/yuriybalandin/cards_detection_project/main/main.py

![download](https://user-images.githubusercontent.com/61317465/150587212-433f53e0-95e4-498f-b54a-07b21fdf74c5.png)

**Целью** являлось: обучить модель, создать веб-интерфейс для ее использования и развернуть веб-интерфейс на сервере.

Решались следующие **задачи**:
1. Поиск данных (был выбран следующий датасет: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/tree/master/images )
2. Предобработка данных и лейблов для выбранной модели - yolov5
3. Обучение модели (использовалось - https://docs.ultralytics.com/)
4. Создание веб интерфейса (использовалась библиотека streamlit)
5. Деплой на серевер (Использовалось облако от streamlit для развертывания их приложений)
6. Радоваться, что ты молодец и все работает 

В папке notebooks находится ноутбук с предобработкой данных и тренировкой модели, а также отчет о получившихся метриках и лоссах (все более чем достойно: precision - 0.95, recall - 0.94).
Более красивую и интерактивную версию отчета с метриками и лоссами можно найти здесь:
https://wandb.ai/yuriy42/YOLOv5/reports/-yolov5--VmlldzoxNDY5MTIy

Приложение можно запустить локально - для этого требуется файл с весами и main.py (в одной папке). Предварительно установив все зависимости (requiremets.txt) необходимо в комнадной строке выполнить: *streamlit run main.py*

Что касается работы модели, то здесь работает магия от yolоv5 - даже при использовании уменьшенной версии, картинки как классифицируются, так и детектируются очень хорошо несмотря на тени/засвеченности и т.п. Визуально боксы отрисовываются почти везде хорошо. Для повышения точности работы можно попробовать добавить больше данных и обучить  старшую версию yolov5.

Относительно полезности полученной модели:
1. Создать робота как партнера для игры в карты - с помощью этой модели робот мог бы распознавать карты и мог решать как играть;
2. Потенциально можно использовать в казино - следить за мошенничесвтом и т.п.;
3. Качество получившейся модели и ее саму можно использовать как подтверждение возможности проводить детекцию кредитных карт, а полученную модель взять за первоначальный бейзлайн.
