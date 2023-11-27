import pandas as pd

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')

# 1. Which was the most ordered item?
chipo['item_name'].value_counts().rename_axis('item').reset_index(name='counts')

# 2. Which was the most ordered item count?
chipo['item_name'].value_counts().rename_axis('item').reset_index(name='counts')[0:1]

# 3. What was the most ordered item in the choice_description column?
chipo['choice_description'].value_counts().rename_axis('item').reset_index(name='counts')[0:1]

# 4. How many items were ordered in total?
sum(chipo['quantity'])

# 5. Turn the item price into a float
# type(chipo['item_price'][0]) // I don't want to take the first element, I want all elements
# chipo['item_price'].dtype // dtype('O') what does this mean?
chipo['item_price'].apply(type).unique() # [str]
chipo['item_price'] = pd.to_numeric(chipo['item_price'].str.replace("$", ""))
chipo['item_price'].apply(type).unique() # [float]

# 6. How much was the revenue for the period in the dataset?
f'${(chipo["item_price"] * chipo["quantity"]).sum():,.2f}'

# 7. How many orders were made in the period?
chipo["order_id"].nunique(dropna=True)

# 8. What is the average revenue amount per order?
f'${(chipo["item_price"] * chipo["quantity"]).sum()/chipo["order_id"].nunique():,.2f}'

# 9. How many different items are sold?
chipo["item_name"].nunique(dropna=True)
