import os
import pandas as pd


folder_path = '/mnt/bit/zmx/data/data_mining/10G_data_new' 

parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]

if not parquet_files:
    print("未找到任何 parquet 文件。")
else:
    print(f"找到 {len(parquet_files)} 个 parquet 文件，开始分析...\n")

# 读取前几个文件进行展示
for idx, file in enumerate(parquet_files[:3]): 
    print(f"📁 文件 {idx+1}: {os.path.basename(file)}")
    try:
        df = pd.read_parquet(file)
        print("🔹 数据类型：\n", df.dtypes)
        print("🔹 前5行数据：\n", df.head())
        print("🔹 第一行数据：\n", df.iloc[0])
    except Exception as e:
        print(f"读取失败：{e}")
    print("\n" + "-"*80 + "\n")
