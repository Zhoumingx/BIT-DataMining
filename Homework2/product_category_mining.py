import pandas as pd
import os
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 设置路径
parquet_dir = './30G_data'
product_catalog_path = './product_catalog.json'

# 加载商品目录
with open(product_catalog_path, 'r', encoding='utf-8') as f:
    product_catalog = json.load(f)
product_map = {item['id']: item['category'] for item in product_catalog['products']}

# 定义商品子类别与其所属大类的映射
category_to_group = {
    # 电子产品
    '智能手机': '电子产品', '笔记本电脑': '电子产品', '平板电脑': '电子产品',
    '智能手表': '电子产品', '耳机': '电子产品', '音响': '电子产品',
    '相机': '电子产品', '摄像机': '电子产品', '游戏机': '电子产品',

    # 服装
    '上衣': '服装', '裤子': '服装', '裙子': '服装', '内衣': '服装',
    '鞋子': '服装', '帽子': '服装', '手套': '服装', '围巾': '服装', '外套': '服装',

    # 食品
    '零食': '食品', '饮料': '食品', '调味品': '食品', '米面': '食品',
    '水产': '食品', '肉类': '食品', '蛋奶': '食品', '水果': '食品', '蔬菜': '食品',

    # 家居
    '家具': '家居', '床上用品': '家居', '厨具': '家居', '卫浴用品': '家居',

    # 办公
    '文具': '办公', '办公用品': '办公',

    # 运动户外
    '健身器材': '运动户外', '户外装备': '运动户外',

    # 玩具
    '玩具': '玩具', '模型': '玩具', '益智玩具': '玩具',

    # 母婴
    '婴儿用品': '母婴', '儿童课外读物': '母婴',

    # 汽车用品
    '车载电子': '汽车用品', '汽车装饰': '汽车用品'
}


# 收集所有订单的大类组合
transactions = []

for file in sorted(os.listdir(parquet_dir)):
    if file.endswith('.parquet') and file.startswith('part-'):
        df = pd.read_parquet(os.path.join(parquet_dir, file), engine='pyarrow')
        for record in df['purchase_history'].dropna():
            try:
                history = json.loads(record) if isinstance(record, str) else record
                item_ids = [item['id'] for item in history.get('items', [])]
                # 获取大类（去重）
                major_groups = list(set(
                    category_to_group.get(product_map.get(item_id)) 
                    for item_id in item_ids 
                    if item_id in product_map and product_map.get(item_id) in category_to_group
                ))
                if major_groups:
                    transactions.append(major_groups)
            except Exception as e:
                continue

# 转换为 one-hot 编码
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# 频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
rules['antecedents'] = rules['antecedents'].apply(set)
rules['consequents'] = rules['consequents'].apply(set)

# 电子产品相关
is_electronics = lambda row: '电子产品' in row['antecedents'] or '电子产品' in row['consequents']
electronics_rules = rules[rules.apply(is_electronics, axis=1)].sort_values(by='support', ascending=False).head(10)

# 非电子产品间典型关联（排除电子产品）
other_rules = rules[~rules.apply(is_electronics, axis=1)].sort_values(by='support', ascending=False).head(10)

# 合并结果（部分展示）
combined_rules = pd.concat([electronics_rules, other_rules])
combined_rules_display = combined_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# 打印终端输出
print("\n📊 电子产品相关 & 非电子产品部分关联规则（前10条）：\n")
print(combined_rules_display.to_string(index=False))

