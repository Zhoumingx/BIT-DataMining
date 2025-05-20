import pandas as pd
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from itertools import combinations

# 设置中文字体
font_path = "/mnt/cfs/bit/zmx/data/Microsoft Yahei.ttf"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  

# 设置路径
parquet_folder = './30G_data'
catalog_path = 'product_catalog.json'

# 加载商品目录
with open(catalog_path, 'r', encoding='utf-8') as f:
    catalog = json.load(f)
product_df = pd.DataFrame(catalog['products'])
product_map = product_df.set_index('id')[['category', 'price']].to_dict('index')

# 商品类别映射
category_mapping = {
    '电子产品': ['智能手机', '笔记本电脑', '平板电脑', '智能手表', '耳机', '音响', '相机', '摄像机', '游戏机'],
    '服装': ['上衣', '裤子', '裙子', '内衣', '鞋子', '帽子', '手套', '围巾', '外套'],
    '食品': ['零食', '饮料', '调味品', '米面', '水产', '肉类', '蛋奶', '水果', '蔬菜'],
    '家居': ['家具', '床上用品', '厨具', '卫浴用品'],
    '办公': ['文具', '办公用品'],
    '运动户外': ['健身器材', '户外装备'],
    '玩具': ['玩具', '模型', '益智玩具'],
    '母婴': ['婴儿用品', '儿童课外读物'],
    '汽车用品': ['车载电子', '汽车装饰'],
}
def map_to_main_category(cat):
    for main_cat, sub_cats in category_mapping.items():
        if cat in sub_cats:
            return main_cat
    return '其他'

# 数据收集容器
monthly_order_counts = defaultdict(int)
monthly_category_counts = defaultdict(lambda: defaultdict(int))  # {month: {category: count}}
user_purchase_sequences = defaultdict(list)  # {user_id: [(timestamp, category)]}

# 遍历数据文件
for file in os.listdir(parquet_folder):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(parquet_folder, file), engine='pyarrow')
        for _, row in df.iterrows():
            try:
                uid = row['id']
                purchase = json.loads(row['purchase_history'])
                purchase_date = pd.to_datetime(purchase['purchase_date'])
                month_str = purchase_date.strftime('%Y-%m')
                items = purchase.get('items', [])

                monthly_order_counts[month_str] += 1

                for item in items:
                    product_info = product_map.get(item['id'])
                    if product_info:
                        main_cat = map_to_main_category(product_info['category'])
                        monthly_category_counts[month_str][main_cat] += 1
                        user_purchase_sequences[uid].append((purchase_date, main_cat))
            except Exception:
                continue

# 构建月度订单量 DataFrame
df_orders = pd.DataFrame(list(monthly_order_counts.items()), columns=['month', 'order_count']).sort_values('month')

# 构建月度商品类别趋势 DataFrame
category_data = []
for month, cats in monthly_category_counts.items():
    for cat, count in cats.items():
        category_data.append((month, cat, count))
df_category_trends = pd.DataFrame(category_data, columns=['month', 'category', 'count'])

# ⏱️ 分析时间顺序模式：先买 A 再买 B
sequence_counts = defaultdict(int)
for seq in user_purchase_sequences.values():
    sorted_seq = sorted(seq, key=lambda x: x[0])
    categories_only = [cat for _, cat in sorted_seq]
    for i in range(len(categories_only) - 1):
        pair = (categories_only[i], categories_only[i + 1])
        sequence_counts[pair] += 1

df_seq = pd.DataFrame(
    [(a, b, c) for (a, b), c in sequence_counts.items() if a != b],
    columns=['Category_A', 'Category_B', 'Count']
).sort_values('Count', ascending=False)

# -----------------------------------
# 📊 可视化部分
# -----------------------------------

# 1. 月度订单量
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_orders, x='month', y='order_count', marker='o')
plt.title('月度订单数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_order_quantity.png', dpi=300)
plt.close()

# 2. 类别购买趋势
plt.figure(figsize=(14, 7))
pivot_trend = df_category_trends.pivot_table(index='month', columns='category', values='count', fill_value=0)
pivot_trend.plot(figsize=(14, 7), marker='o')
plt.title('各商品类别月度购买趋势')
plt.xlabel('月份')
plt.ylabel('购买次数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('product_category_monthly_purchasing.png', dpi=300)
plt.close()

# 3. Top 时序模式
plt.figure(figsize=(10, 6))
top_seq = df_seq.head(10)
sns.barplot(data=top_seq, y=top_seq['Category_A'] + ' → ' + top_seq['Category_B'], x='Count')
plt.title('Top 10 商品类别购买顺序模式')
plt.xlabel('次数')
plt.ylabel('购买顺序')
plt.tight_layout()
plt.savefig('product_category_purchase_order.png', dpi=300)
plt.close()
