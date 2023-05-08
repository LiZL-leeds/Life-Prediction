#!/usr/bin/env pythonS
# coding: utf-8

# In[1]:
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import messagebox

import matplotlib
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

#get_ipython().run_line_magic('matplotlib', 'inline')

from math import sqrt

#from sklearn.datasets._base import load_data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from torch.nn import LSTM
# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[4]:


#用于从数组中删除异常值
def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)

#构建文本序列
def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# 留一评估：一组数据为测试集，其他所有数据全部拿来训练。从数据字典中获取训练和测试数据集
def get_train_test(data_dict, name, window_size=8):
    data_sequence=data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


#RE
def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


#MAE & RMSE
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# In[4]:


Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
dir_path = 'dataset/'
Battery = {}
for name in Battery_list:
    print('Load Dataset ' + name + ' ...')
    path = glob.glob(dir_path + name + '/*.xlsx')
    dates = []
    for p in path:
        df = pd.read_excel(p, sheet_name=1)
        print('Load ' + str(p) + ' ...')
        dates.append(df['Date_Time'][0])
    idx = np.argsort(dates)
    path_sorted = np.array(path)[idx]
    
    count = 0
    discharge_capacities = []
    health_indicator = []
    internal_resistance = []
    CCCT = []
    CVCT = []
    for p in path_sorted:
        df = pd.read_excel(p,sheet_name=1)
        print('Load ' + str(p) + ' ...')
        cycles = list(set(df['Cycle_Index']))
        for c in cycles:
            df_lim = df[df['Cycle_Index'] == c]
            #Charging
            df_c = df_lim[(df_lim['Step_Index'] == 2)|(df_lim['Step_Index'] == 4)]
            c_v = df_c['Voltage(V)']
            c_c = df_c['Current(A)']
            c_t = df_c['Test_Time(s)']
            #CC or CV
            df_cc = df_lim[df_lim['Step_Index'] == 2]
            df_cv = df_lim[df_lim['Step_Index'] == 4]
            CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)']))
            CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)']))

            #Discharging
            df_d = df_lim[df_lim['Step_Index'] == 7]
            d_v = df_d['Voltage(V)']
            d_c = df_d['Current(A)']
            d_t = df_d['Test_Time(s)']
            d_im = df_d['Internal_Resistance(Ohm)']

            if(len(list(d_c)) != 0):
                time_diff = np.diff(list(d_t))
                d_c = np.array(list(d_c))[1:]
                discharge_capacity = time_diff*d_c/3600 # Q = A*h
                discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
                discharge_capacities.append(-1*discharge_capacity[-1])

                dec = np.abs(np.array(d_v) - 3.8)[1:]
                start = np.array(discharge_capacity)[np.argmin(dec)]
                dec = np.abs(np.array(d_v) - 3.4)[1:]
                end = np.array(discharge_capacity)[np.argmin(dec)]
                health_indicator.append(-1 * (end - start))

                internal_resistance.append(np.mean(np.array(d_im)))
                count += 1

    discharge_capacities = np.array(discharge_capacities)
    health_indicator = np.array(health_indicator)
    internal_resistance = np.array(internal_resistance)
    CCCT = np.array(CCCT)
    CVCT = np.array(CVCT)
    
    idx = drop_outlier(discharge_capacities, count, 40)
    df_result = pd.DataFrame({'cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
                              'capacity':discharge_capacities[idx],
                              'SoH':health_indicator[idx],
                              'resistance':internal_resistance[idx],
                              'CCCT':CCCT[idx],
                              'CVCT':CVCT[idx]})
    Battery[name] = df_result


# ### 如果上面的读取数据集失败，可以通过下面的方式加载已提取出来的数据

# In[3]:


Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('dataset/CALCE.npy', allow_pickle=True)
Battery = Battery.item()


# In[5]:


#Rated_Capacity = 1.1
fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
for name,color in zip(Battery_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result['cycle'], df_result['capacity'], color, label='Battery_'+name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 1°C')
plt.legend()
plt.show()


# In[5]:

class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, n_class=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):           # x shape: (batch_size, seq_len, input_size)
        out, _ = self.cell(x) 
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)      # out shape: (batch_size, n_class=1)
        return out


# In[6]:


def tain(lr=0.001, feature_size=16, hidden_dim=128, num_layers=2, weight_decay=0.0, mode = 'LSTM', EPOCH=1000, seed=0):
    score_list, result_list = [], []
    #mae_list, rmse_list, re_list = [], [], []
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size=feature_size)
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(input_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_ = [0], []
        mae, rmse, re = 1, 1, 1
        score_, score = 1,1
        for epoch in range(EPOCH):
            X = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)#(batch_size, seq_len, input_size)
            y = np.reshape(train_y[:,-1]/Rated_Capacity,(-1,1)).astype(np.float32)# shape 为 (batch_size, 1)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output= model(X)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if (epoch + 1)%100 == 0:
                test_x = train_data.copy()    #每100次重新预测一次
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)                # shape: (batch_size, 1, input_size)
                    pred = model(x)
                    next_point = pred.data.numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)                          #测试值加入原来序列用来继续预测下一个点
                    point_list.append(next_point)                     #保存输出序列最后一个点的预测值
                y_.append(point_list)                                 #保存本次预测所有的预测值
                loss_list.append(loss)
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
                print('epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae, rmse, re))
            score = [re, mae, rmse]
            if (loss < 1e-3) and (score_[0] < score[0]):
                break
            score_ = score.copy()

        score_list.append(score_)
        result_list.append(y_[-1])
    return score_list, result_list


def LSTM():
    global mode
    mode = 'LSTM'

def RNN():
    global mode
    mode = 'RNN'
# In[7]:

frameT = Tk()
frameT.geometry('500x200+400+200')
frameT.title('选择需要使用的神经网络')
frame = Frame(frameT)
frame.pack(padx=10, pady=10)  # 设置外边距
frame_1 = Frame(frameT)
frame_1.pack(padx=10, pady=10)  # 设置外边距
frame1 = Frame(frameT)
frame1.pack(padx=10, pady=10)
v1 = StringVar()
v2 = StringVar()

window_size = 128
EPOCH = 1000
lr = 0.001    # learning rate  0.01 epoch 10
mode = None
hidden_dim = 256
num_layers = 2
weight_decay = 0.0
Rated_Capacity = 1.1

btn = Button(frame, width=20, text='RNN', font=("宋体", 14), command=RNN).pack(fil=X, padx=10)
btn_1 = Button(frame_1, width=20, text='LSTM', font=("宋体", 14), command=LSTM).pack(fil=X, padx=10)
etb = Button(frame1, width=10, text='确认', font=("宋体", 14), command=frameT.destroy).pack(fill=Y, padx=10)
frameT.mainloop()


SCORE = []
print_score_list = []
for seed in range(4):
    print('seed: ', seed)
    score_list, _ = tain(lr=lr, feature_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers, 
                         weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)
    print('------------------------------------------------------------------')
    print_score_list = score_list
    for s in score_list:
        SCORE.append(s)

mlist = ['re', 'mae', 'rmse']
for i in range(3):
    s = [line[i] for line in SCORE]
    print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')


# In[18]:


def get_data():
    try:
        cycle_life = int(cycle_entry.get())
        if cycle_life < 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid positive integer")
        return

    # get the relevant data from aa
    remaining_life = aa[(cycle_life + 1)]

    # show the result in the same window
    result_entry.delete(0, tk.END)  # clear any previous value
    result_entry.insert(0, str(remaining_life))

    root.update()  # force update to get the correct window size
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    root.geometry("+%d+%d" % (x, y))  # move the window to the center of the screen



seed = 3
_, result_list = tain(lr, feature_size=window_size, num_layers=num_layers,
                            weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)
for i in range(4):
    name = Battery_list[i]
    train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)

    aa = train_data[:window_size+1].copy() # 第一个输入序列
    [aa.append(a) for a in result_list[i]] # 测试集预测结果

    battery = Battery[name]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(battery['cycle'], battery['capacity'], 'b.', label=name)
    ax.plot(battery['cycle'], aa, 'r.', label='Prediction')
    plt.plot([-1,1000],[Rated_Capacity*0.7, Rated_Capacity*0.7],
             c='black', lw=1, ls='--')  # 临界点直线
    print(print_score_list)
    plt.text(75, 0.6, "re: " + str(print_score_list[i][0]))
    plt.text(75, 0.5, "mae: " + str(print_score_list[i][1]))
    plt.text(75, 0.4, "rmse: " + str(print_score_list[i][2]))
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)',
           title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()
    plt.show(block=False)

    root = tk.Tk()
    root.title("请输入周期数来获取电池RUL")

    # create a label and entry for the cycle life input
    cycle_label = tk.Label(root, text="Enter the cycle: ")
    cycle_label.place(relx=0.4, rely=0.3, anchor=tk.CENTER)
    cycle_entry = tk.Entry(root)
    cycle_entry.place(relx=0.65, rely=0.3, anchor=tk.CENTER)

    # create a label for displaying the result
    result_label = tk.Label(root, text="prediction: ")
    result_label.place(relx=0.4, rely=0.4, anchor=tk.CENTER)
    result_entry = tk.Entry(root)
    result_entry.place(relx=0.65, rely=0.4, anchor=tk.CENTER)

    # create a button to get the data
    get_data_button = tk.Button(root, text="Get Data", command=get_data)
    get_data_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # create a plot
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)',
           title='Capacity degradation at ambient temperature of 1°C')

    # run the main loop
    root.update()  # force update to get the correct window size
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    root.geometry("+%d+%d" % (x, y))  # move the window to the center of the screen
    root.geometry("500x300+0+0")
    root.mainloop()





