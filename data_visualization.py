import seaborn as sns
import matplotlib.pyplot as plt
import function as f

df = f.get_processed_data('train.csv')
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.show()

