"""k-平均法を部分空間法で利用"""
import time
import numpy as np
from tqdm import tqdm


start_time = time.time()


# ローカルに保存したmnistを呼び出し
def load_mnist() -> np.ndarray:
    """ローカルに保存したmnistを呼び出し"""
    train = np.load('train.npz', allow_pickle=True)
    X_train = train['arr_0']
    y_train = train['arr_1']
    test = np.load('test.npz', allow_pickle=True)
    X_test = test['arr_0']
    y_test = test['arr_1']
    return X_train, X_test, y_train, y_test


def classify_by_label(data, target):
    num_label = []
    for c in np.unique(target):
    X = data[np.where(target == str(c))[0]]      # class c のデータを取得
    # N, D = X.shape                              # データ数, データ次元数

    num_label.append(X)

    return num_label


def main():
    """main"""
    # mnistの呼び出し(比較実験時にデータを同じにするため)
    X_train, X_test, y_train, y_test = load_mnist()

    num_label = classify_by_label(X_train, y_train)

    # 処理時間の計算
    elapsed_time = int(time.time() - start_time)
    # 処理時間の表示
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = elapsed_time % 3600 % 60
    print(f"elapsed time : {str(elapsed_hours).zfill(2) + ":"
                            + str(elapsed_minutes).zfill(2) + ":"
                            + str(elapsed_seconds).zfill(2)}")
