import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Load data from CSV file
dataset = pd.read_csv('ShopBasket_Optimisation.csv', header=None)

# Convert data to list of transactions
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, len(dataset.columns))])

# Run Apriori algorithm to find frequent itemsets and association rules
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Convert results to DataFrame for easier analysis and visualization
results = pd.DataFrame(list(rules))

# Extract left-hand side, right-hand side, support, confidence, and lift from the results
results['lhs'] = results['ordered_statistics'].apply(lambda x: list(x[0].items_base))
results['rhs'] = results['ordered_statistics'].apply(lambda x: list(x[0].items_add))

results['support'] = results['support'].astype(float)
results['confidence'] = results['ordered_statistics'].apply(lambda x: getattr(x[0], 'confidence')).astype(float)

results['support'] = results['support'].astype(float)
results['confidence'] = results['ordered_statistics'].apply(lambda x: x[0].confidence).astype(float)
results['lift'] = results['ordered_statistics'].apply(lambda x: x[0].lift).astype(float)


# Print top 10 association rules by lift
print(results.nlargest(n=10, columns='lift'))

# Visualize support vs. confidence using a scatter plot
fig, ax = plt.subplots()
ax.scatter(results['support'], results['confidence'], s=results['lift']*1000, alpha=0.5)
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_title('Association Rules')
for i, txt in enumerate(results['lhs']):
    ax.annotate(', '.join(txt), (results.iloc[i]['support'], results.iloc[i]['confidence']))
for i, txt in enumerate(results['rhs']):
    ax.annotate(', '.join(txt), (results.iloc[i]['support'], results.iloc[i]['confidence']))
plt.show()
