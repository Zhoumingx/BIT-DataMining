import os
import pandas as pd
import re
import time
import json
from datetime import datetime, timedelta

def load_dataset(folder_path):
    all_dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.parquet'):
            file_path = os.path.join(folder_path, fname)
            try:
                df = pd.read_parquet(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        print(f"⚠️ No valid parquet files found in {folder_path}")
        return pd.DataFrame()

def filter_gender_other(df, dataset_name):
    total = len(df)
    gender_other_count = df[df['gender'] == '其他'].shape[0]
    gender_other_ratio = gender_other_count / total if total > 0 else 0

    # 删除 gender 为 "其他" 的行
    df_filtered = df[df['gender'] != '其他']
    new_total = len(df_filtered)

    return df_filtered

def extract_purchase_metrics(row):
    try:
        data = json.loads(row)
        return pd.Series({
            "avg_price": data.get("avg_price", 0),
            "payment_status": data.get("payment_status", ""),
            "category_count": len(set(data.get("categories", "").split(','))) if isinstance(data.get("categories", ""), str) else 0
        })
    except Exception:
        return pd.Series({"avg_price": 0, "payment_status": "", "category_count": 0})

def identify_high_value_users(df):
    # 时间准备
    today = pd.Timestamp.today()
    recent_threshold = today - pd.Timedelta(days=30)

    # 解析 purchase_history 字段
    purchase_info = df['purchase_history'].apply(extract_purchase_metrics)
    df = pd.concat([df, purchase_info], axis=1)

    # 将 last_login 和 registration_date 转为时间格式
    df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce')
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')

    # 高收入阈值（75百分位）
    income_threshold = df['income'].quantile(0.75)

    # 筛选规则
    high_value_mask = (
        (df['income'] >= income_threshold) &
        (df['age'].between(25, 55)) &
        (df['avg_price'] > 5000) &
        (df['payment_status'] == '已支付') &
        (df['is_active'] == True) &
        (df['last_login'] >= recent_threshold)
    )

    high_value_users = df[high_value_mask]

    print(f"🔍 符合高价值用户条件的用户数量：{len(high_value_users)}")
    return high_value_users


path_10g = './10G_data_new'
path_30g = './30G_data_new'

df_10g = load_dataset(path_10g)
df_30g = load_dataset(path_30g)

# 处理两个数据集
df_10g_filtered = filter_gender_other(df_10g, '10G')
df_30g_filtered = filter_gender_other(df_30g, '30G')

start_10g = time.time()
print(start_10g)
high_value_users = identify_high_value_users(df_10g_filtered)
vis_time_10g = time.time() - start_10g
print(f"10G 数据用户分析耗时：{vis_time_10g:.2f} 秒")
high_value_users.to_csv("high_value_users_10G.csv", index=False)

start_30g = time.time()
print(start_30g)
high_value_users = identify_high_value_users(df_30g_filtered)
vis_time_30g = time.time() - start_30g
print(f"30G 数据用户分析耗时：{vis_time_30g:.2f} 秒")
high_value_users.to_csv("high_value_users_30G.csv", index=False)