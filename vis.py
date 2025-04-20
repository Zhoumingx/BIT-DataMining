import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import time

# 设置 seaborn 风格
sns.set(style="whitegrid")

# 设置中文字体
my_font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family'] = my_font.get_name() 

def load_dataset(folder_path):
    dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.parquet'):
            fpath = os.path.join(folder_path, fname)
            try:
                df = pd.read_parquet(fpath)
                dfs.append(df)
            except Exception as e:
                print(f"❌ 读取失败 {fname}: {e}")
    return pd.concat(dfs, ignore_index=True)

def visualize_dataset(df, title_prefix, save_path):
    start_time = time.time()

    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'{title_prefix} 数据集可视化', fontproperties=my_font, fontsize=20)

    # 年龄分布 - 直方图 & 箱线图
    sns.histplot(df['age'], bins=30, kde=True, color='skyblue', ax=axs[0, 0])
    axs[0, 0].set_title('年龄分布（直方图）', fontproperties=my_font)
    axs[0, 0].set_xlabel('年龄', fontproperties=my_font)

    sns.boxplot(x=df['age'], color='lightgreen', ax=axs[0, 1])
    axs[0, 1].set_title('年龄分布（箱线图）', fontproperties=my_font)
    axs[0, 1].set_xlabel('年龄', fontproperties=my_font)

    # 收入分布
    sns.histplot(df['income'], bins=30, kde=True, color='orange', ax=axs[0, 2])
    axs[0, 2].set_title('收入分布', fontproperties=my_font)
    axs[0, 2].set_xlabel('收入', fontproperties=my_font)

    # 性别分布（饼图）
    gender_counts = df['gender'].value_counts()
    axs[1, 0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                 textprops={'fontproperties': my_font}, colors=sns.color_palette("pastel"))
    axs[1, 0].set_title('性别分布', fontproperties=my_font)

    # 国家分布
    country_counts = df['country'].value_counts()
    axs[1, 1].pie(country_counts, labels=country_counts.index, autopct='%1.1f%%',
                 textprops={'fontproperties': my_font}, colors=sns.color_palette("muted"))
    axs[1, 1].set_title('国家分布', fontproperties=my_font)

    # 活跃用户分布
    active_counts = df['is_active'].value_counts()
    labels = ['活跃用户' if val else '非活跃用户' for val in active_counts.index]
    axs[1, 2].pie(active_counts, labels=labels, autopct='%1.1f%%',
                 textprops={'fontproperties': my_font}, colors=['lightcoral', 'lightblue'])
    axs[1, 2].set_title('用户活跃状态分布', fontproperties=my_font)

    # 注册时间分布（按月）
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
    reg_counts = df['registration_date'].dt.to_period('M').value_counts().sort_index()
    reg_counts.plot(kind='bar', color='slateblue', ax=axs[2, 0])
    axs[2, 0].set_title('用户注册时间分布（月）', fontproperties=my_font)
    axs[2, 0].set_xlabel('注册月份', fontproperties=my_font)
    axs[2, 0].set_ylabel('用户数', fontproperties=my_font)

    # 填补空图位
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 已保存图像到 {save_path}")


path_10g = './10G_data_new' 
path_30g = './30G_data_new'  

# 10G 数据处理
start_10g = time.time()
df_10g = load_dataset(path_10g)
load_time_10g = time.time() - start_10g
print(f"加载 10G 数据耗时：{load_time_10g:.2f} 秒")

visualize_dataset(df_10g, title_prefix='10G', save_path='visualization_10G.png')

vis_time_10g = time.time() - start_10g
print(f"可视化 10G 数据耗时：{vis_time_10g:.2f} 秒")

# 30G 数据处理
start_30g = time.time()
df_30g = load_dataset(path_30g)
load_time_30g = time.time() - start_30g
print(f"加载 30G 数据耗时：{load_time_30g:.2f} 秒")

visualize_dataset(df_30g, title_prefix='30G', save_path='visualization_30G.png')

vis_time_30g = time.time() - start_30g
print(f"可视化 30G 数据耗时：{vis_time_30g:.2f} 秒")
