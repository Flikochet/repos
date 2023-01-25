import matplotlib.pyplot as plt
from numpy import array
from numpy import interp
from numpy import linspace
from scipy import interpolate
from scipy import optimize
from statistics import mean
from os import listdir
import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from io import StringIO
from random import randint
import altair as alt

@@ -125,43 +126,47 @@ def file_check(X_list, Y_list, delta_wave):
        return True


def file_read_mode_2(file):
    s = file.readline()
    s = s[s.find('=') + 2:-12]
def file_read_mode_2(str_file):
    rows = str_file.split('\n')
    s = rows[1][rows[1].find('=') + 2:-12]
    Int_Time = float(s.replace(',', '.'))
    file.readline()
    x_list = []
    y_list = []
    for row in file:

    file_rows = rows[2:]

    for row in file_rows:
        if len(row) > 1:
            wave = float(row[0:row.find('	')].replace(',', '.'))
            sample = float(row[row.find('	'):].replace(',', '.'))
            x_list.append(wave)
            y_list.append(sample/Int_Time)

    return [x_list, y_list]


def file_read_mode_1(file):
    file.readline()
    s = file.readline()
    if_ms = s.find('ms')
def file_read_mode_1(str_file):
    rows = str_file.split('\n')
    if_ms = rows[1].find('ms')
    Int_Time = 0
    if if_ms == -1:
        print('Не милисекунды')
        st.write('Не милисекунды')
    else:
        Int_Time = float(s[4 + 3 + if_ms:-2].replace(',', '.'))
        Int_Time = float(rows[1][4 + 3 + if_ms:-2].replace(',', '.'))

    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    #file.readline()
    #file.readline()
    #file.readline()
    #file.readline()
    #file.readline()
    #file.readline()

    x_list = []
    y_list = []

    for row in file:
    file_rows = rows[8:]

    for row in file_rows:
        if len(row) > 1:
            wave = float(row[0:row.find(';')].replace(',', '.'))
            if not row[2 + row.find(';')].isdigit():
@@ -179,23 +184,24 @@ def file_read_mode_1(file):
    return [x_list, y_list]


def open_folder(folder_name, border_1, border_2):
def open_folder(folder_name, files_list):
    delta_wave = 40
    file_name = listdir(folder_name)
    Y = []
    X = []
    n = 0
    dust_amount = 0

    for i in file_name:
        if i.endswith('.TXT') or i.endswith('.txt'):
            n = n + 1
            #print(n)
            file = open(folder_name + '\\' + i)
            if i.endswith('.TXT'):
                buf = file_read_mode_1(file)
            if i.endswith('.txt'):
                buf = file_read_mode_2(file)
    for i in files_list:
        if i.name.endswith('.TXT') or i.name.endswith('.txt'):
            stringio = StringIO(i.getvalue().decode("utf-8"))
            string_data = stringio.read()
            #st.write(stringio)
            #file = i.getvalue()
            #st.write(file)
            #st.write(type(file))
            if i.name.endswith('.TXT'):
                buf = file_read_mode_1(string_data)
            if i.name.endswith('.txt'):
                buf = file_read_mode_2(string_data)
            x_list = buf[0]
            y_list = buf[1]
            max_wave = x_list[max(enumerate(y_list), key=lambda x: x[1])[0]]
@@ -239,28 +245,28 @@ def Write_File(folder_name, title, x, y):

def Absorb_Graph_Menu(number, graph_count):
    st.title('Данные графика ' + str(number + 1))
    dust_level = st.number_input('Введите во сколько раз сигнал должен быть сильнее шума', value=1.5, step=0.1, key=number)
    folder_name = st.text_input('Введите полный путь до папки с файлами',
                          r'',
                                key=1000+number)
    border_1 = st.number_input('Введите нижнюю границу', value=2000, step=10, key=2000+number)
    border_2 = st.number_input('Введите верхнюю границу', value=2500, step=10, key=3000+number)
    raw_data = st.selectbox('Выводить необработанные данные?', (False, True), key=4000+number)
    window = st.number_input('Введите размер окна сглаживания в нм', value=2.0, step=0.1, key=5000+number)
    folder_name = st.text_input('Введите название образца', 'sample', key=number)
    file_list = st.file_uploader('Переместите сюда папку с файлами', accept_multiple_files=True, key=1000+number)
    dust_level = st.number_input('Введите во сколько раз сигнал должен быть сильнее шума', value=1.5, step=0.1,
                                 key=2000+number)
    border_1 = st.number_input('Введите нижнюю границу', value=2000, step=10, key=3000+number)
    border_2 = st.number_input('Введите верхнюю границу', value=2500, step=10, key=4000+number)
    raw_data = st.selectbox('Выводить необработанные данные?', (False, True), key=5000+number)
    window = st.number_input('Введите размер окна сглаживания в нм', value=2.0, step=0.1, key=6000+number)
    figure_list = []
    for i in range(graph_count):
        figure_list.append(i + 1)
    figure_list_OD = []
    for i in range(2 * graph_count):
        figure_list_OD.append(i + 1)
    figure = st.selectbox('В какой координатной сетке вывести график поглощения? Укажите ее номер',
                          np.array(figure_list), index=number, key=6000+number)
    reference_number = st.selectbox('Укажите номер графика референса', np.array(figure_list), index=number, key=7000+number)
    window_OD = st.number_input('Введите размер окна сглаживания для OD', value=2.0, step=0.1, key=8000+number)
    raw_data_OD = st.selectbox('Выводить необработанные данные OD?', (False, True), key=9000 + number)
                          np.array(figure_list), index=number, key=7000+number)
    reference_number = st.selectbox('Укажите номер графика референса', np.array(figure_list), index=number, key=8000+number)
    window_OD = st.number_input('Введите размер окна сглаживания для OD', value=2.0, step=0.1, key=9000+number)
    raw_data_OD = st.selectbox('Выводить необработанные данные OD?', (False, True), key=10000 + number)
    figure_OD = st.selectbox('В какой координатной сетке вывести график OD? Укажите ее номер',
                          np.array(figure_list_OD), index=number, key=10000 + number)
    if_reference = st.selectbox('Строить ли график OD?', (False, True), index=True, key=11000 + number)
                          np.array(figure_list_OD), index=number, key=11000 + number)
    if_reference = st.selectbox('Строить ли график OD?', (False, True), index=True, key=12000 + number)
    out_txt_data = False
    output_folder_name = ''
    #print('Выводить график в txt файл? Введите 1 если да или пропустите')
@@ -270,58 +276,7 @@ def Absorb_Graph_Menu(number, graph_count):
    #    output_folder_name = input()
    return [folder_name, dust_level, window, border_1, border_2,
            raw_data, out_txt_data, output_folder_name, figure, number,
            reference_number, window_OD, figure_OD, raw_data_OD, if_reference]


def OD_Graph_Menu(number):
    delta_wave = 40
    border_1 = 2000
    border_2 = 2500
    out_txt_data = False
    output_folder_name = ''
    print('Введите полный путь до папки с файлами')
    folder_name = input()
    print('Введите во сколько раз сигнал должен быть сильнее шума')
    dust_level = float(input())
    print('Введите размер окна сглаживания для данной кривой в нм')
    window = int(input())
    print('Введите размер окна сглаживания для кривой OD для исследуемых данных в нм')
    window_OD = int(input())
    print('Изменить интересующие границы длин волн? По умолчанию от 2000 до 2500. '
          'Введите 1 если да или пропустите')
    if input() == '1':
        print('Введите нижнюю границу')
        border_1 = float(input())
        print('Введите верхнюю границу')
        border_2 = float(input())
    print('Выводить график OD в txt файл? Введите 1 если да или пропустите')
    if input() == '1':
        out_txt_data = True
        print('Введите полный путь до папки, в которую сохранить файлы с графиком')
        output_folder_name = input()

    return [folder_name, dust_level, window, window_OD, border_1, border_2, out_txt_data, output_folder_name, number]


def Reference_Menu():
    delta_wave = 40
    border_1 = 2000
    border_2 = 2500
    print('Введите полный путь до папки с файлами')
    folder_name = input()
    print('Введите во сколько раз сигнал должен быть сильнее шума')
    dust_level = float(input())
    print('Введите размер окна сглаживания для данной кривой в нм')
    window = int(input())
    print('Изменить интересующие границы длин волн? По умолчанию от 2000 до 2500. '
          'Введите 1 если да или пропустите')
    if input() == '1':
        print('Введите нижнюю границу')
        border_1 = float(input())
        print('Введите верхнюю границу')
        border_2 = float(input())

    return [folder_name, dust_level, window, border_1, border_2]
            reference_number, window_OD, figure_OD, raw_data_OD, if_reference, file_list]


def Smooth(X, Y, window):
@@ -374,8 +329,9 @@ def Plot_Absorb_Graph(input):
    output_folder_name = input[7]
    figure = input[8]
    number = input[9]
    file_list = input[15]

    out = open_folder(folder_name, border_1, border_2)
    out = open_folder(folder_name, file_list)
    imposter_seek(out[0], out[1], dust_level, out[2])
    buf1 = grow_definition(out[0], out[1])
    buf2 = border_def(buf1[0], buf1[1], border_1, border_2)
@@ -394,9 +350,10 @@ def Plot_Absorb_Graph(input):


def title_name(folder_name, depth):
    title = folder_name[:folder_name.rfind('\\')]
    for i in range(depth):
        title = title[title.rfind('\\') + 1:]
    #title = folder_name[:folder_name.rfind('\\')]
    #for i in range(depth):
    #    title = title[title.rfind('\\') + 1:]
    title = folder_name
    return title


@@ -412,22 +369,24 @@ def Plot_OD_Graph(input, input_ref):
    window_OD = input[11]
    figure_OD = input[12]
    raw_data_OD = input[13]
    file_list = input[15]

    folder_name_ref = input_ref[0]
    dust_level_ref = input_ref[1]
    window_ref = input_ref[2]
    border_1_ref = input_ref[3]
    border_2_ref = input_ref[4]
    file_list_ref = input_ref[15]

    out = open_folder(folder_name_ref, border_1_ref, border_2_ref)
    out = open_folder(folder_name_ref, file_list_ref)
    imposter_seek(out[0], out[1], dust_level_ref, out[2])
    buf1 = grow_definition(out[0], out[1])
    buf2 = border_def(buf1[0], buf1[1], border_1, border_2)
    buf2 = border_def(buf1[0], buf1[1], border_1_ref, border_2_ref)
    buf = Smooth(buf2[0], buf2[1], window_ref)
    x_ref = buf[0]
    y_ref = buf[1]

    out = open_folder(folder_name, border_1, border_2)
    out = open_folder(folder_name, file_list)
    imposter_seek(out[0], out[1], dust_level, out[2])
    buf1 = grow_definition(out[0], out[1])
    buf2 = border_def(buf1[0], buf1[1], border_1, border_2)
@@ -452,48 +411,24 @@ def Plot_OD_Graph(input, input_ref):


def Main_Menu():
    graph_count = st.selectbox(
        'Сколько графиков будет на входе?',
        (1, 2, 3, 4, 5, 6, 7, 8))
    graph_count = st.selectbox('Сколько графиков будет на входе?', (1, 2, 3, 4, 5, 6, 7, 8))
    #figure_define = st.selectbox('Вывести все графики в одних координатах?', (True, False))
    calculation = st.button('Провести расчет')
    print('Построить графики поглощения или график OD? Введите 1 если графики поглощения или 2 если график OD')
    #graph_mode = input()
    #for i in range(graph_count):
    #    figures.append(plt.figure(figsize=(10 + i, 6 + i), dpi=180*i))
    #    st.write(figures[i])
    graph_mode = '1'
    if graph_mode == '1':
        print('Сколько графиков поглощения необходимо построить?')
        setup_absorb_graph = []
    setup_absorb_graph = []
    for i in range(graph_count):
        setup_absorb_graph.append(Absorb_Graph_Menu(i, graph_count))

    if calculation:
        for i in range(graph_count):
            setup_absorb_graph.append(Absorb_Graph_Menu(i, graph_count))
        print('Идет работа...')
        if calculation:
            for i in range(graph_count):
                Plot_Absorb_Graph(setup_absorb_graph[i])
                if setup_absorb_graph[i][14]:
                    Plot_OD_Graph(setup_absorb_graph[i], setup_absorb_graph[setup_absorb_graph[i][10] - 1])
            set_list = []
            for i in range(graph_count):
                set_list.append(setup_absorb_graph[i][8])
                set_list.append(setup_absorb_graph[i][12])
            for i in range(len(list(set(set_list)))):
                st.pyplot(plt.figure(i + 1))
    if graph_mode == '2':
        setup_OD_ref = []
        setup_OD = []
        print('Введите сколько будет объектов для исследования (не считая референса)')
        graph_count = int(input())
            Plot_Absorb_Graph(setup_absorb_graph[i])
            if setup_absorb_graph[i][14]:
                Plot_OD_Graph(setup_absorb_graph[i], setup_absorb_graph[setup_absorb_graph[i][10] - 1])
        set_list = []
        for i in range(graph_count):
            print('Введите данные для референса')
            setup_OD_ref.append(Reference_Menu())
            print('Введите данные для исследуемого образца')
            setup_OD.append(OD_Graph_Menu(i))
        print('Идет работа...')
        if calculation:
            for i in range(graph_count):
                Plot_OD_Graph(setup_OD_ref[i], setup_OD[i])
            set_list.append(setup_absorb_graph[i][8])
            set_list.append(setup_absorb_graph[i][12])
        for i in range(len(list(set(set_list)))):
            st.pyplot(plt.figure(i + 1))


def OD_count(x_ref, y_ref, x, y):#расчет самих данных для OD
    x0 = linspace(x[0], x[-1], round(len(x)))
    y1 = interp(x0, x, y)
    y1_ref = interp(x0, x_ref, y_ref)
    OD = []
    for i in range(len(x0)):
        counted = math.log10(y1_ref[i] / y1[i])
        OD.append(counted)
    return [x0, OD]
def grow_definition(x, y):#выравнивание данных по порядку х
    Point = []
    x_list = []
    y_list = []
    for i in range(len(x)):
        Point.append([x[i], y[i]])
        Point.sort()
    for i in range(len(Point)):
        x_list.append(Point[i][0])
        y_list.append(Point[i][1])
    return [x_list, y_list]
Main_Menu()
plt.show()
#st.set_page_config(layout="wide", initial_sidebar_state="auto")
