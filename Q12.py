import pandas as pd

sales_data = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'A', 'B', 'A'],
    'QuantitySold': [10, 15, 8, 12, 20, 18, 5, 25, 22, 30]
})

product_totals = {}

for index, row in sales_data.iterrows():
    product = row['Product']
    quantity_sold = row['QuantitySold']

    if product in product_totals:
        product_totals[product] += quantity_sold
    else:
        product_totals[product] = quantity_sold

sorted_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)

top_5_products = sorted_products[:5]

print("Top 5 Products Sold the Most:")
for product, total_quantity_sold in top_5_products:
    print(f"{product}: {total_quantity_sold} units")
