import glob
import os
from pathlib import Path

import pandas as pd
from config import *

resule_dir_path = os.path.join(BASE_PATH, 'result/')
table_dir_path = os.path.join(resule_dir_path, 'table/')
param_dir_path = os.path.join(resule_dir_path, 'param/')
Path(param_dir_path).mkdir(parents=True, exist_ok=True)

table_file_path_list = sorted(glob.glob(table_dir_path + '*'))

time = []
setting = []
command = []
result = []
velocity = []
torque = []
loop_correction = []

for file_path in table_file_path_list:
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))

    df = pd.read_csv(file_path)

    time.append(file_name[:-6])
    setting.append(df.iloc[0].tolist()[1:])
    command.append(df.iloc[1].tolist()[1:])
    result.append(df.iloc[2].tolist()[1:])
    velocity.append(df.iloc[3].tolist()[1:])
    torque.append(df.iloc[4].tolist()[1:])
    loop_correction.append(df.iloc[5].tolist()[1:])

pd.DataFrame(index=time, columns=df.columns[1:], data = setting).to_csv(param_dir_path+'setting.csv')
pd.DataFrame(index=time, columns=df.columns[1:], data = command).to_csv(param_dir_path+'command.csv')
pd.DataFrame(index=time, columns=df.columns[1:], data = result).to_csv(param_dir_path+'result.csv')
pd.DataFrame(index=time, columns=df.columns[1:], data = velocity).to_csv(param_dir_path+'velocity.csv')
pd.DataFrame(index=time, columns=df.columns[1:], data = torque).to_csv(param_dir_path+'torque.csv')
pd.DataFrame(index=time, columns=df.columns[1:], data = loop_correction).to_csv(param_dir_path+'loop_correction.csv')

