import pandas as pd
import json
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 设置中文字体
font_path = "/mnt/cfs/bit/zmx/data/Microsoft Yahei.ttf"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  

# --- 1. 配置路径 ---
parquet_folder = './30G_data'
catalog_path = 'product_catalog.json'

# --- 2. 加载商品目录 + 商品类别映射 ---
with open(catalog_path, 'r', encoding='utf-8') as f:
    catalog = json.load(f)
product_df = pd.DataFrame(catalog['products'])
product_map = product_df.set_index('id')[['category', 'price']].to_dict('index')

# 商品类别映射表（小类到大类）
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

# --- 3. 提取退款分析所需的交易数据 ---
def extract_refund_transactions(df):
    transactions = []
    for record in df['purchase_history'].dropna():
        try:
            purchase = json.loads(record)
            payment_status = purchase.get('payment_status', '')
            if payment_status in ['已退款', '部分退款']:
                items = purchase.get('items', [])
                categories = set()
                for item in items:
                    item_info = product_map.get(item['id'])
                    if item_info:
                        main_cat = map_to_main_category(item_info['category'])
                        categories.add(main_cat)

                if categories:
                    # 交易项：商品类别 + 状态标签
                    transaction = list(categories) + [f'状态:{payment_status}']
                    transactions.append(transaction)

        except Exception as e:
            continue
    return transactions

# --- 4. 遍历读取所有 parquet 数据 ---
all_transactions = []
for file in os.listdir(parquet_folder):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(parquet_folder, file), engine='pyarrow')
        transactions = extract_refund_transactions(df)
        all_transactions.extend(transactions)

# --- 5. One-hot 编码 ---
te = TransactionEncoder()
te_ary = te.fit(all_transactions).transform(all_transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# --- 6. 挖掘频繁项集 ---
frequent_itemsets = apriori(df_trans, min_support=0.005, use_colnames=True)

# --- 7. 挖掘关联规则（支付状态为后件）---
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)
# refund_rules = rules[rules['consequents'].apply(lambda x: any('状态:已退款' in x or '状态:部分退款' in x))]
# 筛选包含退款状态的关联规则
refund_rules = rules[rules['consequents'].apply(
    lambda x: '状态:已退款' in x or '状态:部分退款' in x
)]


# 打印前几条规则
print(refund_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))

# 可视化前10条 Lift 最大的规则
plt.figure(figsize=(10, 6))
top_rules = refund_rules.sort_values(by='lift', ascending=False).head(10)
labels = top_rules['antecedents'].astype(str) + ' → ' + top_rules['consequents'].astype(str)
sns.barplot(x=top_rules['lift'], y=labels)
plt.title('导致退款的高影响商品组合（Top 10 by Lift）')
plt.xlabel('Lift')
plt.ylabel('规则')
plt.tight_layout()
plt.savefig('refund_category.png', dpi=300)
plt.close()

