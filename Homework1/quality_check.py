import os
import pandas as pd
import re
import time

def is_all_chinese(text):
    return isinstance(text, str) and all('\u4e00' <= ch <= '\u9fff' for ch in text)

def is_valid_email(email):
    return isinstance(email, str) and bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

def is_valid_username(username):
    return isinstance(username, str) and bool(re.match(r"^[A-Za-z0-9_]+$", username))

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

def check_id_unique_in_parquet(folder_path):
    results = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.parquet'):
            file_path = os.path.join(folder_path, fname)
            try:
                df = pd.read_parquet(file_path)
                is_int = pd.api.types.is_integer_dtype(df['id'])
                is_unique = df['id'].is_unique
                results.append((fname, is_int, is_unique))
            except Exception as e:
                results.append((fname, False, False))
    return results

def data_quality_check(df, dataset_name):
    report = {}

    # 字段存在性检查
    report['missing_values'] = df.isnull().sum()

    # user_name 合法性
    report['invalid_user_name'] = df[~df['user_name'].apply(is_valid_username)].shape[0]

    # fullname 合法性（中文）
    report['invalid_fullname'] = df[~df['fullname'].apply(is_all_chinese)].shape[0]

    # email 合法性
    report['invalid_email'] = df[~df['email'].apply(is_valid_email)].shape[0]

    # 汇总打印
    print(f"\n📋 数据集【{dataset_name}】字段合法性检查：")
    print("缺失值统计：")
    print(report['missing_values'])
    print(f"user_name 非法数量：{report['invalid_user_name']}")
    print(f"fullname 非中文数量：{report['invalid_fullname']}")
    print(f"email 非法数量：{report['invalid_email']}")

    return report

def check_dataset_quality(folder_path, df, name):
    # 检查 id 在每个 parquet 文件中是否唯一
    id_check = check_id_unique_in_parquet(folder_path)
    print(f"\n📂 数据集【{name}】的每个 parquet 文件中 id 唯一性检查：")
    for fname, is_int, is_unique in id_check:
        print(f"{fname} - id 类型整数: {is_int}，唯一性: {is_unique}")

    # 其他字段检查
    _ = data_quality_check(df, dataset_name=name)

def filter_gender_other(df, dataset_name):
    total = len(df)
    gender_other_count = df[df['gender'] == '其他'].shape[0]
    gender_other_ratio = gender_other_count / total if total > 0 else 0

    print(f"\n📊 数据集【{dataset_name}】删除前总记录数: {total}")
    print(f"🔍 'gender' 为 '其他' 的记录数: {gender_other_count}，占比: {gender_other_ratio:.2%}")

    # 删除 gender 为 "其他" 的行
    df_filtered = df[df['gender'] != '其他']
    new_total = len(df_filtered)

    print(f"🧹 删除后记录数: {new_total}，减少了: {total - new_total} 条\n")

    return df_filtered

path_10g = './10G_data_new'
path_30g = './30G_data_new'

df_10g = load_dataset(path_10g)
df_30g = load_dataset(path_30g)

start_10g = time.time()
check_dataset_quality(path_10g, df_10g, name='10G')
vis_time_10g = time.time() - start_10g
print(f"10G 数据质量检测耗时：{vis_time_10g:.2f} 秒")

start_30g = time.time()
check_dataset_quality(path_30g, df_30g, name='30G')
vis_time_30g = time.time() - start_30g
print(f"30G 数据质量检测耗时：{vis_time_30g:.2f} 秒")