import pandas as pd

# Load stock data from CSV file (replace 'your_stock_data.csv' with the actual file name)
file_path = "/path/to/your_stock_data.csv"
stock_data = pd.read_csv("/Users/lakshminarayanamandi/Downloads/Movies/FODS/data.csv")

# Display the first few rows of the stock data
print(stock_data.head())

# Extract the closing prices from the stock data
closing_prices = stock_data['ClosingPrice']

# Calculate the variability of stock prices
price_mean = closing_prices.mean()
price_std_dev = closing_prices.std()

# Display insights into stock price movements
print(f"Mean closing price: ${price_mean:.2f}")
print(f"Standard deviation of closing prices: ${price_std_dev:.2f}")
print(f"Minimum closing price: ${closing_prices.min():.2f}")
print(f"Maximum closing price: ${closing_prices.max():.2f}")

# Identify days with significant price movements (e.g., greater than two standard deviations)
significant_price_movements = stock_data[abs(stock_data['ClosingPrice'] - price_mean) > 2 * price_std_dev]
print("\nDays with significant price movements:")
print(significant_price_movements)
