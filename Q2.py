import pandas as pd

order_data = pd.DataFrame({
    'Customer_ID': [1, 1, 2, 2, 3],
    'Order_Date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03', '2023-01-02'],
    'Product_Name': ['Product_A', 'Product_B', 'Product_A', 'Product_C', 'Product_B'],
    'Order_Quantity': [2, 3, 1, 2, 4]
})

# 1. The total number of orders made by each customer.
total_orders_by_customer = order_data.groupby('Customer_ID').size().reset_index(name='Total_Orders')
print("Total number of orders made by each customer:")
print(total_orders_by_customer)

# 2. The average order quantity for each product.
avg_order_quantity_per_product = order_data.groupby('Product_Name')['Order_Quantity'].mean().reset_index(name='Average_Order_Quantity')
print("\nAverage order quantity for each product:")
print(avg_order_quantity_per_product)

# 3. The earliest and latest order dates in the dataset.
earliest_order_date = order_data['Order_Date'].min()
latest_order_date = order_data['Order_Date'].max()
print(f"\nEarliest order date: {earliest_order_date}")
print(f"Latest order date: {latest_order_date}")
