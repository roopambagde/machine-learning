import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('ShopBasket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
from apyori import apriori
rules = apriori(transactions=transactions, 
                min_support=0.003, 
                min_confidence=0.2, 
                min_lift=3, 
                min_length=2, 
                max_length=2)
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
results = list(rules)
results_df = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])
top_10 = results_df.nlargest(n=10, columns='Support')
print(top_10)
plt.bar(top_10['Product 1'] + ' - ' + top_10['Product 2'], top_10['Support'])
plt.xticks(rotation=90)
plt.xlabel('Product Pairs')
plt.ylabel('Support')
plt.title('Top 10 Product Pairs by Support')
plt.show()
