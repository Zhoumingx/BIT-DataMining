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
    # 初筛
    income_threshold = df['income'].quantile(0.75)
    high_value_mask_1 = (
        (df['income'] >= income_threshold) &
        (df['age'].between(25, 55)) &
        (df['is_active'] == True)
    )
    df_filtered = df[high_value_mask_1].copy()

    df_filtered['last_login'] = pd.to_datetime(df_filtered['last_login'], errors='coerce', utc=True)

    # 对筛选后的数据进行 JSON 字段解析
    purchase_info = df_filtered['purchase_history'].apply(extract_purchase_metrics)
    df_filtered = pd.concat([df_filtered, purchase_info], axis=1)

    # 精筛
    login_2025_mask = df_filtered['last_login'].dt.year == 2025
    high_value_mask_2 = (
        (df_filtered['avg_price'] > 5000) &
        (df_filtered['payment_status'] == '已支付') &
        (login_2025_mask)
    )

    high_value_users = df_filtered[high_value_mask_2]
    print(f"🏆 最终识别的高价值用户数：{len(high_value_users)}")

    return high_value_users


path_10g = './10G_data_new'
path_30g = './30G_data_new'

df_10g = load_dataset(path_10g)
df_30g = load_dataset(path_30g)

# 处理两个数据集
df_10g_filtered = filter_gender_other(df_10g, '10G')
df_30g_filtered = filter_gender_other(df_30g, '30G')

start_10g = time.time()
# print(start_10g)
high_value_users = identify_high_value_users(df_10g_filtered)
vis_time_10g = time.time() - start_10g
print(f"10G 数据用户分析耗时：{vis_time_10g:.2f} 秒")
high_value_users.to_csv("high_value_users_10G.csv", index=False)

start_30g = time.time()
# print(start_30g)
high_value_users = identify_high_value_users(df_30g_filtered)
vis_time_30g = time.time() - start_30g
print(f"30G 数据用户分析耗时：{vis_time_30g:.2f} 秒")
high_value_users.to_csv("high_value_users_30G.csv", index=False)