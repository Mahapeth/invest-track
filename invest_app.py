import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import statistics
import datetime
import dash_bootstrap_components as dbc
import plotly.express as px
from urllib.parse import urlparse
from urllib.parse import parse_qs
pd.options.mode.chained_assignment = None

data = pd.read_csv('C:/Users/think/Desktop/ВКР/data.csv')
data['al_bdt'] = pd.to_datetime(data['al_bdt'])
data['al_edt'] = pd.to_datetime(data['al_edt'])
data['report_id'] = data['report_id'].astype(np.int64)
data['al_mt'] = data['al_mt'].astype(np.int64)

# Обработка данных для построения графиков по количеству входов в систему
df_for_plot = data[['al_program', 'al_bdt']]
df_for_plot = df_for_plot.iloc[df_for_plot['al_program'].index[df_for_plot['al_program'] == '/index.php'].tolist()]
df_for_plot = df_for_plot.reset_index(drop=True)
df_for_plot['day'] = df_for_plot['al_bdt'].dt.dayofweek
df_for_plot['hour'] = df_for_plot['al_bdt'].dt.hour

for i in range(len(df_for_plot)):
    df_for_plot['day'][i] = df_for_plot['day'][i] + 1

df_for_plot_copy = df_for_plot.copy()

# Обработка данных для построения круговой диаграммы по IP адресам
lst_ip = (data['al_ip'].unique()).tolist()  # Список уникальных IP-адресов
lst_ip.sort()  # Сортировка списка уникальных IP-адресов
lst_ip = lst_ip[1:]  # Исключение пустого элемента

first_el_external_lst = []  # Список первых элементов внешних ip-адресов
for i in range(len(lst_ip)):
    strr = lst_ip[i].split('.')
    first_el_external_lst.append(strr[0])
first_el_external_lst = list(set(first_el_external_lst))
first_el_external_lst.remove('10')
first_el_external_lst.remove('172')
first_el_external_lst.remove('192')
external_ip_lst = []  # Список внешних ip-адресов
for i in range(len(first_el_external_lst)):
    for j in range(len(lst_ip)):
        if (lst_ip[j].split('.'))[0] == first_el_external_lst[i]:
            external_ip_lst.append(lst_ip[j])

ip = list(set(lst_ip) - set(external_ip_lst))

# Обработка данных для построения графика по входам в меню верхнего уровня
df_qs_copy = pd.read_csv('C:/Users/think/Desktop/ВКР/qs_data.csv')

un_arid = df_qs_copy['arid'].value_counts().index.tolist()
arid_len = []
for i in range(len(un_arid)):
    arid_len.append(len(df_qs_copy[df_qs_copy['arid'] == un_arid[i]]))
# Данные для анализа активности пользователей
data_for_plot = pd.read_csv('C:/Users/think/Desktop/ВКР/data_for_plot.csv')
names_columns = data_for_plot.columns.tolist()
count_columns = []
for i in range(len(names_columns)):
    if i % 2 == 1:
        count_columns.append(names_columns[i])

data['suser_id'] = data['suser_id'].astype(str)
unique_users = data['suser_id'].unique().tolist()[1:]
df_index = data.iloc[data['al_program'].index[data['al_program'] == '/index.php'].tolist()]
df_index = df_index.reset_index(drop=True)
df_index['day_week'] = df_index['al_bdt'].dt.dayofweek
df_index['day'] = df_index['al_bdt'].dt.date
df_index['time'] = df_index['al_bdt'].dt.time
for i in range(len(df_index)):
    df_index['day_week'][i] = df_index['day_week'][i] + 1
count_index_of_users = []
index_of_users = []
for i in range(len(unique_users)):
    count_index_of_users.append(len(df_index['suser_id'].index[df_index['suser_id'] == unique_users[i]].tolist()))
    index_of_users.append(df_index['suser_id'].index[df_index['suser_id'] == unique_users[i]].tolist())

df_plot_js = data[['al_ip', 'al_js', 'al_bdt', 'al_edt']]
df_plot_js['day'] = df_plot_js['al_bdt'].dt.date
unique_day_ip = df_plot_js['day'].unique().tolist()
ip_day_lst = []
for i in range(len(unique_day_ip)):
    ip_day_lst.append(df_plot_js.iloc[df_plot_js['day'].index[df_plot_js['day'] == unique_day_ip[i]].tolist()])

all_len_js_ip_lst = []
all_js_ip_lst = []
all_ip_lst = []
for i in range(len(ip_day_lst)):
    new_df = ip_day_lst[i]
    unique_ip = new_df['al_ip'].unique().tolist()
    ip_lst = []
    for j in range(len(unique_ip)):
        ip_lst.append(new_df.loc[new_df['al_ip'].index[new_df['al_ip'] == unique_ip[j]].tolist()])
    all_ip_lst.append(ip_lst)
    js_ip_lst = []
    len_js_ip_lst = []
    for z in range(len(ip_lst)):
        len_js_ip_lst.append(len(ip_lst[z]['al_js'].unique().tolist()))
        js_ip_lst.append(ip_lst[z]['al_js'].unique().tolist())
    all_len_js_ip_lst.append(len_js_ip_lst)
    all_js_ip_lst.append(js_ip_lst)

max_all_js_ip_lst = []
avg_all_js_ip_lst = []
for i in range(len(all_len_js_ip_lst)):
    max_all_js_ip_lst.append(max(all_len_js_ip_lst[i]))
    avg_all_js_ip_lst.append(statistics.mean(all_len_js_ip_lst[i]))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

SIDESTYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'padding': '2rem 1rem',
    'background-color': '#222222',
}

CONTSTYLE = {
    'margin-left': '16rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
    'background-color': '#F7F7F7',
}

card = {
    'margin-bottom': '24px',
    'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)',
}

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(
        [
            html.H2('Инвест', className='display-3', style={'color': 'white'}),
            html.Hr(style={'color': 'white'}),
            dbc.Nav(
                [
                    dbc.NavLink('Анализ активности в системе', href='/', active='exact'),
                    dbc.NavLink('Анализ активности пользователей', href='/page2', active='exact'),
                ],
                vertical=True, pills=True),
        ],
        style=SIDESTYLE,
    ),
    html.Div(id='page-content', children=[], style=CONTSTYLE)
], style={'backgroundColor': '#F7F7F7'})

# Построение графиков на первой странице
df_hist_1 = pd.DataFrame({'День недели': df_for_plot_copy['day'].unique().tolist(),
                          'Количество': df_for_plot['day'].value_counts(sort=False).tolist()})
df_hist_2 = pd.DataFrame({'Время работы': df_for_plot['hour'].unique().tolist(),
                          'Количество': df_for_plot['hour'].value_counts(sort=False).tolist()})
df_hist_2_2 = pd.DataFrame({'Время работы': df_for_plot['hour'].unique().tolist(),
                            'Количество': df_for_plot['hour'].value_counts(sort=False).tolist()})
max_df = pd.DataFrame({'Дата': unique_day_ip, 'Максимальное значение': max_all_js_ip_lst,
                       'Среднее значение': avg_all_js_ip_lst})
avg_df = pd.read_csv('C:/Users/think/Desktop/ВКР Петухова МЛ/Данные/avg_df.csv')

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')])
def pagecontent(pathname):
    if pathname == '/':
        return [
            html.Div([html.Div([html.Div(dcc.Graph(id='graph1', figure=px.bar(df_hist_1, x='День недели',
                                y='Количество',
                               title='Количество входов в систему по дням недели',
                            color='Количество', template='plotly_white', width=600, height=500).update_xaxes(labelalias=
                    {'1':'Понедельник', '2':'Вторник', '3':'Среда', '4':'Четверг', '5':'Пятница', '6':'Суббота',
                     '7':'Воскресенье'}, tickangle=-45), style=card)),
                      html.Div(dcc.Graph(id='graph2', figure=px.bar(df_hist_2, x='Время работы', y='Количество',
                                         title='Количество входов в систему по часам',
                                         color='Количество',
                                         template='plotly_white', width=600, height=500).add_traces(
                          px.line(df_hist_2_2, x='Время работы', y='Количество').data), style=card))],
                                style={'display': 'flex'}), html.Div(dcc.Graph(id='max-graph',
                                                                               figure=px.bar(max_df, x='Дата',
y=['Максимальное значение', 'Среднее значение'], title='Количество сессий на человека в день', template='plotly_white',
barmode='group', color_discrete_sequence=['#dd80cc', '#d4b0f5'], labels=dict(value='Значение',
                                                                             variable='Значение')).add_traces(
           px.line(max_df, x='Дата', y='Максимальное значение',
       color_discrete_sequence=['#dd80cc']).data).add_traces(px.line(max_df, x='Дата', y='Среднее значение',
                                                                     color_discrete_sequence=['#d4b0f5']).data),
                                                                               style=card)), html.Div(
                dcc.Graph(id='avg-graph',
        figure=px.bar(avg_df, x='Дата',
y=['Максимальное значение', 'Среднее значение'], title='Количество просмотров на человека в день', template='plotly_white',
barmode='group', color_discrete_sequence=['#dd80cc', '#d4b0f5'], labels=dict(value='Значение',
                                                                             variable='Значение')).add_traces(
           px.line(avg_df, x='Дата', y='Максимальное значение',
       color_discrete_sequence=['#dd80cc']).data).add_traces(px.line(avg_df, x='Дата', y='Среднее значение',
                                                                     color_discrete_sequence=['#d4b0f5']).data),
                          style=card)),
                      html.Div([html.Div(dcc.Graph(id='pie1', figure=px.pie(
                        pd.DataFrame({'Метод': ['GET', 'POST', "' '"], 'Количество': [len(data[data['al_m'] == 'GET']),
                                                                                      len(data[data['al_m'] == 'POST']),
                                                                                      len(data[data['al_m'] == "' '"])]}
                                     ),
              values='Количество', names='Метод', title='Методы, используемые в системе',
              color_discrete_sequence=px.colors.diverging.Tropic_r, template='plotly_white', width=600, height=500,
               ), style=card)),
                                                  html.Div(dcc.Graph(id='pie2', figure=px.pie(
    pd.DataFrame({'IP-адресс': ['Внутренние', 'Внешние'], 'Количество': [len(ip), len(external_ip_lst)]}),
    values='Количество', names='IP-адресс', title='Количество IP-адресов, входящих в систему',
    color_discrete_sequence=px.colors.diverging.Tropic_r, template='plotly_white', width=600, height=500,
                                                      ), style=card))],
                                                 style={'display': 'flex'}), html.Div([html.Div([
                html.Label('Выбор меню первого уровня', id='label-txt1', style={'font-weight': 'bold'}),
                dcc.Dropdown(id='y-axis-dropdown',
                             options=[{'label': i, 'value': i} for i in un_arid], value='Договоры', multi=False)],
                style={'width': '33%', 'display': 'inline-block', 'font-family': 'Times New Roman',
                       'margin-left': '40px'}),
                 html.Div([html.Label('Выбор характерисктики', id='label-txt2', style={'font-weight': 'bold'}),
                           dcc.Dropdown(id='x-axis-dropdown',
                     options=[{'label': i, 'value': i} for i in count_columns],
                     value='Количество входов в "Договоры"', multi=False)],
                style={'width': '33%', 'display': 'inline-block', 'font-family': 'Times New Roman',
                       'margin-left': '220px'})],
                style={'display': 'flex'}), html.P(), html.Div([html.Div(dcc.Graph(id='graph3',
         figure=px.bar(pd.DataFrame({'Меню верхнего уровня': un_arid, 'Количество': arid_len}),
                    x='Количество', y='Меню верхнего уровня', title='Количество входов в меню верхнего уровня',
                    color='Количество', category_orders={'Меню верхнего уровня': un_arid},
                    template='plotly_white', width=600, height=500), style=card)), html.Div(dcc.Graph(id='histogram',
                                                                                                      style=card))],
                                                 style={'display': 'flex'})])
        ]
    elif pathname == '/page2':
        return [
            html.Div([
                html.Label('Выбор пользователя', id='label-user', style={'font-weight': 'bold',
                                                                         'fontFamily': 'Times New Roman'}),
                dcc.Dropdown(id='user-dropdown',
                             options=[{'label': i, 'value': i} for i in unique_users],
                             placeholder='Выберите пользователя',
                             multi=False)], style={'width': '33%', 'fontFamily': 'Times New Roman'}), html.P(),
            html.P(id='stats-user', style={'fontFamily': 'Times New Roman'}), html.P(id='stats-user2',
                                                                    style={'fontFamily': 'Times New Roman'}),
            html.Div([html.Div(dcc.Graph(id='graph-pie', style=card)),
                                                         html.Div(dcc.Graph(id='graph-pie2', style=card))],
                                                        style={'display': 'flex'}),
            html.Div(dcc.Graph(id='histogram2', style=card)), html.Div(dcc.Graph(id='histogram3', style=card))]

@app.callback(
    Output('histogram', 'figure'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'))
def hist(xaxis_column_name, yaxis_column_name):
    fig = px.bar(data_for_plot[[xaxis_column_name, yaxis_column_name]], x=xaxis_column_name, y=yaxis_column_name,
                 color=xaxis_column_name, title='Количество входов в меню второго уровня', template='plotly_white',
                 width=600, height=500)
    fig.layout.coloraxis.colorbar.title = 'Количество'
    return fig

@app.callback(
    Output('stats-user', 'children'),
    Input('user-dropdown', 'value'))
def correction_text(user_name):
    if user_name != None:
        df = data.iloc[data['suser_id'].index[data['suser_id'] == unique_users[unique_users.index(user_name)]].tolist()]
        df = df.reset_index(drop=True)
        def get_qs_for_data():
            lst_qs = []  # Список словарей по всем url
            for i in range(len(df['al_url'])):
                u = urlparse(df['al_url'][i])  # Разбивка url на компоненты
                u = parse_qs(u.query)  # Словарь. Ключи словаря — это уникальные имена переменных запроса,
                # а значения — это списки значений для каждого имени
                vv = []  # Список значений словаря
                for v in u.values():
                    vv.append(v[0])
                kk = []  # Список ключей словаря
                for k in u:
                    kk.append(k)
                ur = {}
                for i in range(len(vv)):
                    ur[kk[i]] = vv[i]
                lst_qs.append(ur)
            return lst_qs
        h = pd.DataFrame(get_qs_for_data())
        h['rpid'].unique().tolist()
        kv = len((h.iloc[h['rpid'].index[h['rpid'] == '20000000000000437'].tolist()])['JS'].unique().tolist())
        if kv == 0:
            return 'Подбирал ли менеджер квартиры клиентам: Нет'
        else:
            return f'Количество раз подбора квартир: {kv}'

@app.callback(
    Output('stats-user2', 'children'),
    Input('user-dropdown', 'value'))
def correction_text(user_name):
    if user_name != None:
        df = data.iloc[data['suser_id'].index[data['suser_id'] == unique_users[unique_users.index(user_name)]].tolist()]
        df = df.reset_index(drop=True)
        def get_qs_for_data():
            lst_qs = []  # Список словарей по всем url
            for i in range(len(df['al_url'])):
                u = urlparse(df['al_url'][i])  # Разбивка url на компоненты
                u = parse_qs(u.query)  # Словарь. Ключи словаря — это уникальные имена переменных запроса,
                # а значения — это списки значений для каждого имени
                vv = []  # Список значений словаря
                for v in u.values():
                    vv.append(v[0])
                kk = []  # Список ключей словаря
                for k in u:
                    kk.append(k)
                ur = {}
                for i in range(len(vv)):
                    ur[kk[i]] = vv[i]
                lst_qs.append(ur)
            return lst_qs
        h = pd.DataFrame(get_qs_for_data())
        h['rpid'].unique().tolist()
        reserv = len((h.iloc[h['rpid'].index[h['rpid'] == '18500000000000107'].tolist()])['JS'].unique().tolist())
        if reserv == 0:
            return 'Резервировал ли менеджер квартиры клиентам: Нет'
        else:
            return f'Количество раз резерва квартир: {reserv}'

@app.callback(
    Output('graph-pie', 'figure'),
    Input('user-dropdown', 'value'))
def pie1(user_name):
    if user_name == None:
        return px.bar(width=600, height=500)
    else:
        df = data.iloc[data['suser_id'].index[data['suser_id'] == unique_users[unique_users.index(user_name)]].tolist()]
        df = df.reset_index(drop=True)
        lst_ip_user = df['al_ip'].unique().tolist()  # Список уникальных IP-адресов
        lst_ip_user.sort()  # Сортировка списка уникальных IP-адресов
        first_el_external_lst_user = []  # Список первых элементов внешних ip-адресов
        for i in range(len(lst_ip_user)):
            strr_user = lst_ip_user[i].split('.')
            first_el_external_lst_user.append(strr_user[0])
        first_el_external_lst_user = list(set(first_el_external_lst_user))
        new_first_el_external_lst_user = []
        for i in range(len(first_el_external_lst_user)):
            if first_el_external_lst_user[i] != '10' and first_el_external_lst_user[i] != '172' and \
                    first_el_external_lst_user[i] != '192':
                new_first_el_external_lst_user.append(first_el_external_lst_user[i])
        external_ip_lst_user = []  # Список внешних ip-адресов
        for i in range(len(new_first_el_external_lst_user)):
            for j in range(len(lst_ip_user)):
                if (lst_ip_user[j].split('.'))[0] == new_first_el_external_lst_user[i]:
                    external_ip_lst_user.append(lst_ip_user[j])
        ip_user = list(set(lst_ip_user) - set(external_ip_lst_user))
        fig = px.pie(
    pd.DataFrame({'IP-адресс': ['Внутренние', 'Внешние'], 'Количество': [len(ip_user), len(external_ip_lst_user)]}),
    values='Количество', names='IP-адресс', title='Количество IP-адресов, c которых входил пользователь',
    color_discrete_sequence=px.colors.diverging.Tropic_r, template='plotly_white', width=600, height=500)
        return fig

@app.callback(
    Output('graph-pie2', 'figure'),
    Input('user-dropdown', 'value'))
def pie1(user_name):
    if user_name == None:
        return px.bar(width=600, height=500)
    else:
        df = data.iloc[data['suser_id'].index[data['suser_id'] == unique_users[unique_users.index(user_name)]].tolist()]
        df = df.reset_index(drop=True)
        methods = df['al_m'].unique().tolist()
        methods_lst = []
        for i in range(len(methods)):
            methods_lst.append(len(df['al_m'].index[df['al_m'] == methods[i]].tolist()))
        fig = px.pie(pd.DataFrame({'Метод': methods, 'Количество': methods_lst}),
    values='Количество', names='Метод', title='Методы, используемые пользователем',
    color_discrete_sequence=px.colors.diverging.Tropic_r, template='plotly_white', width=600, height=500)
        return fig

@app.callback(
    Output('histogram2', 'figure'),
    Input('user-dropdown', 'value'))
def hist2(user_name):
    if user_name == None:
        return px.bar()
    else:
        user = df_index.iloc[index_of_users[unique_users.index(user_name)]]
        user = user.reset_index(drop=True)
        day = user['day'].unique().tolist()
        day_lst = []
        for i in range(len(day)):
            day_lst.append(len(user['day'].index[user['day'] == day[i]].tolist()))
        df_hist_m = pd.DataFrame({'Дата': day, 'Количество': day_lst})
        fig = px.bar(df_hist_m, x='Дата', y='Количество', title='Количество входов в систему по дням',
                     color='Количество', template='plotly_white')
        return fig

@app.callback(
    Output('histogram3', 'figure'),
    Input('user-dropdown', 'value'))
def hist3(user_name):
    if user_name == None:
        return px.bar()
    else:
        df = data.iloc[data['suser_id'].index[data['suser_id'] == unique_users[unique_users.index(user_name)]].tolist()]
        df['day'] = df['al_bdt'].dt.date
        df = df.reset_index(drop=True)
        days_user = df['day'].unique().tolist()
        days_user_lst = []
        for j in range(len(days_user)):
            days_user_lst.append(df['day'].index[df['day'] == days_user[j]].tolist())
        d = []
        for z in range(len(days_user_lst)):
            dd = []
            for k in range(len(days_user_lst[z])):
                if df['al_program'][days_user_lst[z][k]] == '/login.php':
                    dd.append(days_user_lst[z][k])
            dd.append(days_user_lst[z][-1])
            d.append(dd)
        time = []
        for c in range(len(d)):
            time_lst = []
            for cc in range(len(d[c]) - 1):
                try:
                    time_lst.append(str((df.iloc[d[c][cc]:d[c][cc + 1]])['al_edt'][d[c][cc + 1] - 1] -
                                        (df.iloc[d[c][cc]:d[c][cc + 1]])['al_bdt'][d[c][cc]]).split('days ')[1])
                except:
                    time_lst.append(str((df.iloc[d[c][cc]])['al_edt'] - (df.iloc[d[c][cc]])['al_bdt']).split(' ')[1])
            time.append(time_lst)
        sum_time = []
        for l in range(len(d)):
            sum_time_lst = []
            for ll in range(len(d[l]) - 1):
                try:
                    sum_time_lst.append((df.iloc[d[l][ll]:d[l][ll + 1]])['al_edt'][d[l][ll + 1] - 1] -
                                        (df.iloc[d[l][ll]:d[l][ll + 1]])['al_bdt'][d[l][ll]])
                except:
                    sum_time_lst.append((df.iloc[d[l][ll]])['al_edt'] - (df.iloc[d[l][ll]])['al_bdt'])
            sum_time.append(sum_time_lst)
        ss_time = []
        for h in range(len(sum_time)):
            ss_time.append(sum(sum_time[h], datetime.timedelta(0, 0)))
        for i in range(len(ss_time)):
            ss_time[i] = round(
                ((ss_time[i].total_seconds()) // 3600) * 60 + ((ss_time[i].total_seconds() % 3600) // 60) + (
                            (ss_time[i].total_seconds() % 3600) % 60) / 3600, 2)
        df_hist1 = pd.DataFrame({'Дата': days_user, 'Количество': ss_time})
        fig=px.bar(df_hist1, x='Дата', y='Количество', title='Проведенное время (мин) в системе по дням',
               color='Количество', template='plotly_white')
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)