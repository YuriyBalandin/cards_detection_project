import sys
from streamlit import cli as stcli

from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static

import tempfile

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import json

import torch
import cv2

from stqdm import stqdm

import tkinter as tk
from tkinter import filedialog

@st.cache()
def load_model(path='yolov5/runs/train/exp6/weights/best.pt'):
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    return detection_model


def detect_image(image, model):
    pred = model(image)
    pred_df = pred.pandas().xyxy[0].sort_values('confidence', ascending=False)
    pred_image = pred.render()[0]
    if pred_df.shape[0] > 0:
        if pred_df.confidence.iloc[0] > 0.5:
            return pred_image, pred_df.name.iloc[0]
        else:
            return pred_image, 'Неизвестное животное'
    else:
        return pred_image, 'Нет животного'



def main():

    st.title('Детекция игральных карт')

    model = load_model('best.pt')


    st.header('Обработка фото')
    file = st.file_uploader('Загрузите изображение')
    if file:
        image = np.array(Image.open(file))
        image, pred = detect_image(image, model)
        st.header('Результаты распознавания')
        st.metric('Вид', pred)
        st.image(image)



if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
