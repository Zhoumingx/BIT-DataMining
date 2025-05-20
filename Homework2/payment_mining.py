import pandas as pd
import json
import os
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# 设置中文字体
font_path = "/mnt/cfs/bit/zmx/data/Microsoft Yahei.ttf"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  

# 设置文件路径
parquet_folder = './30G_data'
catalog_path = './product_catalog.json'

# 加载商品目录
with open(catalog_path, 'r', encoding='utf-8') as f:
    catalog = json.load(f)
product_df = pd.DataFrame(catalog['products'])
product_map = product_df.set_index('id')[['category', 'price']].to_dict('index')

# 定义商品大类映射
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

# 将小类映射到大类
def map_to_main_category(cat):
    for main_cat, sub_cats in category_mapping.items():
        if cat in sub_cats:
            return main_cat
    return '其他'

# 提取数据的函数
def extract_transactions(df):
    transactions = []
    high_value_payment_methods = []

    for record in df['purchase_history'].dropna():
        try:
            purchase = json.loads(record)
            payment_method = purchase.get('payment_method')
            items = purchase.get('items', [])
            categories = set()

            for item in items:
                item_info = product_map.get(item['id'])
                if item_info:
                    main_cat = map_to_main_category(item_info['category'])
                    categories.add(main_cat)
                    if item_info['price'] > 5000:
                        high_value_payment_methods.append(payment_method)

            if categories and payment_method:
                # 支付方式放入 transactions 作为先验项
                transaction = [payment_method] + list(categories)
                transactions.append(transaction)

        except Exception:
            continue

    return transactions, high_value_payment_methods

# 读取所有 parquet 文件
all_transactions = []
high_value_methods = []

for file in os.listdir(parquet_folder):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(parquet_folder, file), engine='pyarrow')
        transactions, hv_methods = extract_transactions(df)
        all_transactions.extend(transactions)
        high_value_methods.extend(hv_methods)

# 转为 one-hot 编码 DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(all_transactions).transform(all_transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# 挖掘频繁项集
frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)

# 提取关联规则
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)

# 支付方式列表（即 catalog 中未出现的条目，推断为支付方式）
all_categories = {map_to_main_category(p['category']) for p in catalog['products']}
def is_payment_to_category_rule(row):
    return (
        len(row['antecedents']) == 1 and 
        list(row['antecedents'])[0] not in all_categories and 
        all(cat in all_categories for cat in row['consequents'])
    )

# 保留支付方式 => 商品类别的规则
valid_rules = rules[rules.apply(is_payment_to_category_rule, axis=1)]

# 打印部分规则
print(valid_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))

# 统计高价值商品支付方式分布
hv_payment_df = pd.DataFrame({'payment_method': high_value_methods})
top_hv_payments = hv_payment_df['payment_method'].value_counts(normalize=True)

# 可视化频繁规则

# 选取前10条规则（按置信度降序）
top_rules = valid_rules.sort_values(by='confidence', ascending=False).head(10).copy()

# 美化标签：去除 frozenset，并拼接箭头
def format_rule_label(row):
    antecedent = '、'.join(row['antecedents'])
    consequent = '、'.join(row['consequents'])
    return f'{antecedent} → {consequent}'

top_rules['rule_label'] = top_rules.apply(format_rule_label, axis=1)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_rules,
    x='confidence',
    y='rule_label',
    palette='Blues_d'
)
plt.title('Top 10 支付方式 → 商品类别 关联规则（按置信度排序）')
plt.xlabel('置信度（Confidence）')
plt.ylabel('规则')
plt.tight_layout()
plt.savefig('top10_category_to_payment_rules.png', dpi=300)
plt.close()

# 可视化高价值商品首选支付方式
plt.figure(figsize=(8, 5))
top_hv_payments.head(10).plot(kind='bar')
plt.title('高价值商品支付方式分布')
plt.ylabel('占比')
plt.xlabel('支付方式')
plt.tight_layout()
plt.savefig('high_value_payment_preferences.png', dpi=300)
plt.close()