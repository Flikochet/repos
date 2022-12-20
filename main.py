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
from random import randint
import altair as alt


###fig = plt.figure()
#fig = plt.figure()
###ax = fig.add_subplot(2, 1, 1)

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def Stream_Gui():
    button_2 = st.button('Провести расчет')
    return [button_2]

def rolling_window(a, win):
    window = win
    if window == 0: window = 2
    if window % 2 == 1: window += 1
    out = []
    for i in range(round(window / 2)):
        sum = 0
        for n in range(round(i*2) + 1):
            sum = sum + a[n]
        sum = sum / (2 * i + 1)
        out.append(sum)
    for i in range(len(a) - window):
        sum = 0
        for n in range(window):
            sum = sum + a[i + n]
        sum = sum / window
        out.append(sum)
    for i in range(round(window / 2)):
        v = round(window / 2) - i + 1
        sum = 0
        for n in range(v * 2 + 1):
            sum = sum + a[len(a)-(v*2) + n - 1]
        sum = sum / (2 * v + 1)
        out.append(sum)
    out[0] = a[0]
    out[1] = a[1]
    out[2] = a[2]
    out[-3] = a[-3]
    out[-2] = a[-2]
    out[-1] = a[-1]
    return out


def imposter_seek(x_list, y_list, dust_level, dust_amount):
    y_border = dust_level * dust_amount
    n = - 1
    for i in range(len(x_list)):
        n = n + 1
        if y_list[n] < y_border:
            del y_list[n]
            del x_list[n]
            n = n - 1


def dust_definition(y_list):
    return(mean(y_list))


def dust_destroy(y_list):
    z = mean(y_list)
    for i in range(len(y_list)):
        y_list[i] = y_list[i] - z


def border_def(x_list, y_list, border_1, border_2):
    nom = -1
    for i in range(len(y_list)):
        nom = nom + 1
        if x_list[nom] < border_1 or x_list[nom] > border_2:
            del x_list[nom]
            del y_list[nom]
            nom = nom - 1
    return [x_list, y_list]


def min(sequence):
    low = sequence[0] # need to start with some value
    for i in sequence:
        if i < low:
            low = i
    return low


def file_check(X_list, Y_list, delta_wave):
    x_list = X_list.copy()
    y_list = Y_list.copy()
    num_max = max(enumerate(y_list), key=lambda x: x[1])[0]
    y1 = y_list[num_max]
    del y_list[num_max]
    num = max(enumerate(y_list), key=lambda x: x[1])[0]
    y2 = y_list[num]

    y3 = min(y_list)
    num = y_list.index(y3)
    del y_list[num]
    y4 = min(y_list)
    #print(y1 / mean(y_list))
    #print(x_list[num_max] - delta_wave)
    #print(x_list[num_max] + delta_wave)
    #print(x_list[0], x_list[-1])
    if ((y1 / mean(y_list)) < (2)) or ((x_list[num_max] - delta_wave) < x_list[0]) or ((x_list[num_max] + delta_wave) > x_list[-1]):
        return False
    else:
        return True


def file_read_mode_2(file):
    s = file.readline()
    s = s[s.find('=') + 2:-12]
    Int_Time = float(s.replace(',', '.'))
    file.readline()
    x_list = []
    y_list = []
    for row in file:
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
    Int_Time = 0
    if if_ms == -1:
        print('Не милисекунды')
    else:
        Int_Time = float(s[4 + 3 + if_ms:-2].replace(',', '.'))

    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()

    x_list = []
    y_list = []

    for row in file:
        if len(row) > 1:
            wave = float(row[0:row.find(';')].replace(',', '.'))
            if not row[2 + row.find(';')].isdigit():
                row = row[2 + row.find(';'):]
            else:
                row = row[1 + row.find(';'):]
            sample = float(row[0:row.find(';')].replace(',', '.'))
            if not row[2 + row.find(';')].isdigit():
                row = row[2 + row.find(';'):]
            else:
                row = row[1 + row.find(';'):]
            dark = float(row[0:row.find(';')].replace(',', '.'))
            x_list.append(wave)
            y_list.append((sample - dark) / Int_Time)
    return [x_list, y_list]


def open_folder(folder_name, border_1, border_2):
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
            x_list = buf[0]
            y_list = buf[1]
            max_wave = x_list[max(enumerate(y_list), key=lambda x: x[1])[0]]
            max_y = y_list[max(enumerate(y_list), key=lambda x: x[1])[0]]
            # plt.plot(x_list, y_list, color=(0.8, 0.8, 0.8))

            # dust_destroy(y_list)
            #border_def(x_list, y_list, border_1, border_2)

            if file_check(x_list, y_list, delta_wave):
                max_wave = x_list[max(enumerate(y_list), key=lambda x: x[1])[0]]
                x = array(x_list)
                y = array(y_list)
                f = interpolate.interp1d(x, y, kind=3)
                x1 = linspace(max_wave - delta_wave, max_wave + delta_wave, 200)

                solution = optimize.minimize_scalar(lambda x: -f(x),
                                                    bounds=[max_wave - delta_wave, max_wave + delta_wave],
                                                    method='bounded')
                sol = str(solution)
                sol = sol[3 + sol.find('x'):]
                # plt.plot(x1, f(x1), color=(0.8, 0.8, 0.8))

                X.append(float(sol))
                Y.append(float(f(float(sol))))

            else:
                dust_amount = dust_definition(y_list)
                # X.append(max_wave)
                # Y.append(max_y)
    #border_def(X, Y, border_1, border_2)
    output = [X, Y, dust_amount]
    return output


def Write_File(folder_name, title, x, y):
    file = open(folder_name + '\\' + str(title) + '.txt', 'w')
    for i in range(len(x)):
        file.write(str(x[i]) + '	' + str(y[i]) + '\n')


def Absorb_Graph_Menu(number, graph_count):
    st.title('Данные графика ' + str(number + 1))
    dust_level = st.number_input('Введите во сколько раз сигнал должен быть сильнее шума', value=1.5, step=0.1, key=number)
    folder_name = st.text_input('Введите местоположение папки',
                          r'C:\Users\relop\PycharmProjects\Fotonic\25.10.2022\25.10.2022\sample2\log_25окт22_171251',
                                key=1000+number)
    border_1 = st.number_input('Введите нижнюю границу', value=2000, step=10, key=2000+number)
    border_2 = st.number_input('Введите верхнюю границу', value=2500, step=10, key=3000+number)
    raw_data = st.selectbox('Выводить необработанные данные?', (False, True), key=4000+number)
    window = st.number_input('Введите размер окна сглаживания в нм', value=2.0, step=0.1, key=5000+number)
    figure_list = []
    for i in range(graph_count):
        figure_list.append(i + 1)
    figure = st.selectbox('В какой координатной сетке вывести график поглощения? Укажите ее номер', np.array(figure_list), index=number, key=6000+number)
    reference_number = st.selectbox('Укажите номер графика референса', np.array(figure_list), index=number, key=7000+number)
    window_OD = st.number_input('Введите размер окна сглаживания для OD', value=2.0, step=0.1, key=8000+number)
    out_txt_data = False
    output_folder_name = ''
    #print('Выводить график в txt файл? Введите 1 если да или пропустите')
    #if input() == '1':
    #    out_txt_data = True
    #    print('Введите полный путь до папки, в которую сохранить файлы с графиком')
    #    output_folder_name = input()
    return [folder_name, dust_level, window, border_1, border_2,
            raw_data, out_txt_data, output_folder_name, figure, number, reference_number, window_OD]


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


def Smooth(X, Y, window):
    x = array(X)
    y = array(Y)

    x1 = linspace(X[0], X[-1], 10 * round(len(X)))
    y1 = interp(x1, x, y)

    delta = (x1[-1] - x1[0]) / len(x1)
    win = round(window / delta)

    yhat = rolling_window(y1, win)

    # yhat = savgol_filter(y1, 51, 10) # window size 51, polynomial order 3
    return [x1, yhat]


def Plot(X, Y, title, figure):
    plt.figure(figure, figsize=(10,8), dpi=1)
    #plt.figure(figure)
    x1 = X
    yhat = Y

    ###plt.plot(x1, yhat, label=title)

    #title = folder_name[:folder_name.rfind('\\')]

    #plt.title(title, fontsize=30)
    plt.xlabel('\u03BB, нм')
    plt.ylabel('OD, у.е.')


    plt.plot(x1, yhat, label=title)
    plt.legend(loc='lower left')

    #st.pyplot(plt.figure(figure))




def Plot_Absorb_Graph(input):
    folder_name = input[0]
    dust_level = input[1]
    window = input[2]
    border_1 = input[3]
    border_2 = input[4]
    raw_data = input[5]
    out_txt_data = input[6]
    output_folder_name = input[7]
    figure = input[8]
    number = input[9]

    title = folder_name[:folder_name.rfind('\\')]
    title = title[title.rfind('\\') + 1:]
    title = title[title.rfind('\\') + 1:]

    out = open_folder(folder_name, border_1, border_2)
    imposter_seek(out[0], out[1], dust_level, out[2])
    buf1 = grow_definition(out[0], out[1])
    buf2 = border_def(buf1[0], buf1[1], border_1, border_2)
    #if raw_data:
    #    plt.plot(buf2[0], buf2[1], color=(0, 0, 1), label='Оригинальные данные')
    if raw_data:
        Plot(buf2[0], buf2[1], 'Оригинальные данные ' + title, figure)
    buf = Smooth(buf2[0], buf2[1], window)
    x = buf[0]
    y = buf[1]

    Plot(x, y, title, figure)
    st.line_chart(pd.DataFrame(y, index=x, columns=['График №' + str(number + 1)]))
    if out_txt_data:
        Write_File(output_folder_name, str(number), x, y)


def Plot_OD_Graph(input_ref, input):
    folder_name = input[0]
    dust_level = input[1]
    window = input[2]
    window_OD = input[3]
    border_1 = input[4]
    border_2 = input[5]
    out_txt_data = input[6]
    output_folder_name = input[7]
    number = input[8]

    folder_name_ref = input_ref[0]
    dust_level_ref = input_ref[1]
    window_ref = input_ref[2]
    border_1_ref = input_ref[3]
    border_2_ref = input_ref[4]

    out = open_folder(folder_name_ref, border_1_ref, border_2_ref)
    imposter_seek(out[0], out[1], dust_level_ref, out[2])
    buf1 = grow_definition(out[0], out[1])
    buf2 = border_def(buf1[0], buf1[1], border_1, border_2)
    buf = Smooth(buf2[0], buf2[1], window_ref)
    x_ref = buf[0]
    y_ref = buf[1]

    out = open_folder(folder_name, border_1, border_2)
    imposter_seek(out[0], out[1], dust_level, out[2])
    buf1 = grow_definition(out[0], out[1])
    buf2 = border_def(buf1[0], buf1[1], border_1, border_2)
    buf = Smooth(buf2[0], buf2[1], window)
    x = buf[0]
    y = buf[1]

    buf1 = OD_count(x_ref, y_ref, x, y)
    buf = Smooth(buf1[0], buf1[1], window_OD)

    title = folder_name[:folder_name.rfind('\\')]
    title = title[title.rfind('\\') + 1:]
    #Plot_Smooth(buf[0], buf[1], title)
    if out_txt_data:
        Write_File(output_folder_name, str(number), buf[0], buf[1])



def Main_Menu():
    graph_count = st.selectbox(
        'Сколько графиков будет на входе?',
        (1, 2, 3, 4, 5, 6, 7, 8))
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
        for i in range(graph_count):
            setup_absorb_graph.append(Absorb_Graph_Menu(i, graph_count))
        print('Идет работа...')
        if calculation:
            for i in range(graph_count):
                Plot_Absorb_Graph(setup_absorb_graph[i])
            set_list = []
            for i in range(graph_count):
                set_list.append(setup_absorb_graph[i][8])
            for i in range(len(list(set(set_list)))):
                st.pyplot(plt.figure(i + 1))
    if graph_mode == '2':
        setup_OD_ref = []
        setup_OD = []
        print('Введите сколько будет объектов для исследования (не считая референса)')
        graph_count = int(input())
        for i in range(graph_count):
            print('Введите данные для референса')
            setup_OD_ref.append(Reference_Menu())
            print('Введите данные для исследуемого образца')
            setup_OD.append(OD_Graph_Menu(i))
        print('Идет работа...')
        if calculation:
            for i in range(graph_count):
                Plot_OD_Graph(setup_OD_ref[i], setup_OD[i])


def OD_count(x_ref, y_ref, x, y):#расчет самих данных для OD
    x0 = linspace(x[0], x[-1], round(len(x)) * 10)
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












