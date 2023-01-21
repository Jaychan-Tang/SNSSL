import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
EXCEL_PATH_FIRST = r'../paper/data/floodnet/6651setxgb-seed'
EXCEL_PATH_SECOND = r'.xlsx'
OUTPUT_PATH = r''
seeds = [0, 1, 2, 3, 4]
files_path = []


for seed in seeds:
    path = EXCEL_PATH_FIRST + str(seed) + EXCEL_PATH_SECOND
    files_path.append(path)

e_file = pd.ExcelFile(files_path[0])
all_sheet = e_file.sheet_names
data_dict = dict()
for sheet in all_sheet:
    data_dict.update({str(sheet): 0})

for path in files_path:
    e_file = pd.ExcelFile(path)
    all_sheet = e_file.sheet_names
    dataframe = pd.read_excel(path, sheet_name=all_sheet)
    # print(dataframe.iloc[0])
    # d_series1 = dataframe.iloc[1]
    # d1 = pd.DataFrame(d_series1)
    lines = dataframe.values
    # for line in lines:
    #     print(line)
    
    for sheet in all_sheet:
        sheet_df = dataframe[sheet]
        lines = sheet_df.values
        for i, line in enumerate(lines):
            if i==len(lines)-1:
                total = data_dict[str(sheet)]
                # print(path[-12:-1], line)
                print(path[-12:-4], sheet, line[-3])
                total += line[-3]
                # print(total)
                data_dict.update({str(sheet):total})
    # print(line[2])
data_list = list()
for key in data_dict:
    # print(key)
    data_dict[key] /= len(seeds)
    data_list.append(data_dict[key])
    
data_array = np.asarray(data_list)
print(data_array)
data_array = data_array.reshape(8, 8)
print(data_array)
data_image = plt.imshow(data_array, cmap='viridis')
x = [0,1,2,3,4,5,6,7]
Nlabels = [1,2,3,4,5,6,7,8]
y = [0,1,2,3,4,5,6,7]
Tlabels = [2,4,6,8,10,12,14,16]
plt.xticks(ticks=x, labels=Tlabels)
plt.yticks(ticks=y, labels=Nlabels)
plt.rc('font', family='Times New Roman')
plt.xlabel("T", fontsize=12)
plt.ylabel("N", fontsize=12)
plt.ylim(-0.5, 7.5)
plt.xlim(-0.5, 7.5)
cb1 = plt.colorbar(data_image, pad=0.005)
plt.show()
