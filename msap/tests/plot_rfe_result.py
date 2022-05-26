from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from msap.modeling.model_evaluation.statistics import get_selected_features
from msap.utils.plot import plot_rfe_line


if __name__ == "__main__":
    data = load_iris(as_frame=True)
    X, y = data['data'][data['target'] < 2], data['target'][data['target'] < 2]
    clf = LogisticRegression(
        # penalty='none',
        random_state=42)

    rfe_result = get_selected_features(
        clf,
        X,
        y,
        5)
    plot_rfe_line(rfe_result)
