import json
from utils import *

# 打开并加载json文件中的数据
with open('list_data.json', 'r') as f:
    loaded_data = json.load(f)

draw_pic(loaded_data['loss1'], loaded_data['loss2'], loaded_data['losses1'], loaded_data['losses2'], loaded_data['losses3'], loaded_data['losses4'],
         "BASE MODEL1", "BASE MODEL2", "BASE DRO MODEL1", "BASE DRO MODEL2", "FINAL DRO MODEL1", "FINAL DRO MODEL2",
         name="ttest")