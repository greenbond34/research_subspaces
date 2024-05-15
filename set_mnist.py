"""mnistをローカルに保存
gitのブランチで実行しようとすると，
mnistの情報量が多すぎて
プッシュできなくなる"""
import ssl
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context


def set_mnist():
    """mnistをローカルに保存"""
    data, target = fetch_openml('mnist_784', version=1, return_X_y=True)
    data = np.array(data)
    label = np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.16,
                                                        random_state=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    np.savez('train.npz', X_train, y_train)
    np.savez('test.npz', X_test, y_test)

    print(data.size)


set_mnist()
