from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_tsne(X, y, random_state=42):
    """
    """
    X = MinMaxScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=random_state).fit_transform(X)

    X_tsne = TSNE(
        n_components=2,
        perplexity=50,
        init='pca',
        random_state=random_state,
        n_jobs=-1,
    ).fit_transform(X)

    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y.values,
    )
    plt.show()
    # print(X_tsne)


def plot_tsne(X_tsne, y, title=None, path_save=None):
    """
    """
    pass


if __name__ == '__main__':
    data = pd.read_csv("msap/tests/data/approach_2/all.csv")
    data = data.set_index('pairID')

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = pd.DataFrame(
        imputer.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

    X = data.drop(['exact_translation_1_1_0_0'], axis=1)
    y = data['exact_translation_1_1_0_0']

    X_tsne = get_tsne(X, y)
    # print(y)
