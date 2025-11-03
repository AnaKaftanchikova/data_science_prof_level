import matplotlib.pyplot as plt
import seaborn as sns

def add_scatter(x, y, title="Scatter Plot"):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.grid(True)
    plt.show()


def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Корреляционная матрица")
    plt.show()
