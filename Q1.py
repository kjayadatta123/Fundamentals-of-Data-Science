item_prices=[]
item_quantities=[]
item_count=int(input("Enter the number of quantities : "))
for i in range(item_count):
    item_prices.append(float(input(f"Enter the price {i+1} :")))
for i in range(item_count):
    item_quantities.append(int(input(f"Enter the quantity {i+1}:")))
sub_total=0
for i in range(item_count):
    sub_total+=item_prices[i]*item_quantities[i]
discount=int(input("Enter the discount :"))
tax=int(input("Enter the tax: "))
discounted_price=(discount/100)*sub_total
total_after_discount=sub_total-discounted_price
tax_price=(tax/100)*total_after_discount
final_price=total_after_discount+tax_price
print("Final Price ₹ :",final_price)
