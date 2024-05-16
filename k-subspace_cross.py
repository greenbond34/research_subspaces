"""plot {Recognition_rate * dimension} (subspace classifier)
elapsed time : 02:01:51 (k:5, dim:40)"""
from pydoc import splitdoc
from re import A, split
import time
import ssl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


start_time = time.time()
ssl._create_default_https_context = ssl._create_unverified_context

count_projection = 0


# mnistをローカルに保存
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


def split_train_data(X_train, y_train, k):
    """学習サンプルをｋ組に分ける"""
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    return np.array_split(X_train, k), np.array_split(y_train, k)


def make_subspace(data, dim) -> np.ndarray:
    """データから部分空間を作成

    Args:
        data (_type_): データ
        dim (_type_): 次元削減後の次元数

    Returns:
        np.ndarray: 部分空間
    """
    N, D = len(data), len(data[0])                             # データ数, データ次元数
    for i in range(D):
        if len(np.unique(data[:, i])) == 1:
            data[np.random.randint(0, N), i] = 1     # おまじない（分散が0になるのを防ぐ）
    correction = np.corrcoef(data.T)               # 自己相関行列を求める
    lam, vec_e = np.linalg.eigh(correction)     # 固有値と固有ベクトルを求める
    vec_e = vec_e[:, np.argsort(lam)[::-1]]     # 固有値の降順に固有ベクトルを並べ替える
    subspace = vec_e[:, :dim]
    return subspace


def clustering_by_subspace(test_data, subspaces, k, dim) -> np.ndarray:
    """k個の部分空間に学習データを写像し、近い空間ごとにクラスタリング
    """
    counter = np.zeros(k)
    norm_list = np.empty(k)
    # hoge = np.empty((k))
    pred_list = [[] for _ in range(k)]
    for v, data in enumerate(test_data):
        for i in range(len(data)):
            for j in range(k):
                norm = 0
                for m in range(dim):
                    norm += np.dot(data[i, :], subspaces[j, :, m])**2  # 各クラスの空間との距離を計算(ユークリッド)
                norm_list[j] = norm
            pred_list[np.argmax(norm_list)].append(data[i])  # 距離が最も近い軸を予測とする
                # np.append(hoge[np.argmax(norm_list)], data[i])
            if v == np.argmax(norm_list):
                counter[v] += 1
        counter[v] /= len(data)
        print(f'counter{v}: {counter[v]}')
    # return np.array(pred_list, dtype=object), counter
    for i in range(len(pred_list)):
        pred_list[i] = np.array(pred_list[i])
    return pred_list, counter


# 部分空間法で予測
def predict(data_test, subspaces, dim) -> np.ndarray:
    """部分空間法で予測する"""
    num_data = len(data_test)
    num_classes = len(subspaces)
    print('Start Prediction')

    pred_list = np.empty(num_data, dtype=int)
    norm_list = np.empty(num_classes)

    for i in tqdm(range(num_data)):
        for j, subspace in enumerate(subspaces):
            norm_k = 0
            for k in range(len(subspace)):
                # 各クラスの空間との距離を計算（ユークリッド）
                norm = sum(np.dot(data_test[i, :], subspace[k, :, v]) ** 2 for v in range(dim))
                if norm > norm_k:
                    norm_k = norm
            norm_list[j] = norm_k
        # 距離が最も近いクラスを予測クラスとする
        pred_list[i] = np.argmax(norm_list)
    print('Prediction Completed')
    return pred_list


# 部分空間法の精度を算出
def accuracy(data_test, target_test, subspaces, dim) -> float:
    """部分空間法の精度を算出"""
    num_data = len(data_test)
    pred_list = predict(data_test, subspaces, dim)
    count = 0
    for i in range(num_data):
        if pred_list[i] == int(target_test[i]):
            count += 1
    acc = count / num_data
    print('accuracy is: ', acc)
    return acc


def main():
    """main"""
    data_cross, target_cross = fetch_openml('mnist_784', version=1, return_X_y=True)

    bunkatu = 5 + 1
    dimdim = 10 + 1
    threshold = 0.90

    # plotのために学習サンプルをk組に分けた時のrec_rateとdimensionを保管する
    store_rec_rate = []
    store_dimension = []

    max_rec_rate = []
    max_dim = []
    k_array = []

    for k in range(1, bunkatu, 2):

        k_array.append(k)

        # 認識率と次元の数値を次元ごとに保存する
        rec_rate = []
        dimension = []

        for dim in range(1, dimdim, 3):

            dimension.append(dim)
            print(f"k={k} {dim}次元")

            for divi in range(5):

                store_pred = []

                data_cross = np.array(data_cross)
                X_train, X_test, y_train, y_test = train_test_split(data_cross, target_cross, test_size=0.16, random_state=divi)
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                data, target = X_train, y_train
                all_class_subspaces = []
                for label in np.unique(target):
                    print(f'class: {label}')
                    # while前の定義
                    label_data, label_target = X_train[np.where(y_train == str(label))[0]], y_train[np.where(y_train == str(label))[0]]
                    train_data, train_target = split_train_data(label_data, label_target, k=k)
                    flag = True
                    loop_counter = 0
                    while flag:
                        loop_counter += 1
                        print(f'loop: {loop_counter}')
                        # make subspaces
                        subspaces = np.empty((k, len(X_train[0]), dim))  # (k, data_dim, reduced_dim)
                        for i, data in enumerate(train_data):
                            subspaces[i] = make_subspace(data, dim=dim)                   # k組目のデータから部分空間を作成
                        # Project the data onto the subspace
                        pred_list, counter = clustering_by_subspace(train_data, subspaces, k, dim)
                        # print([len(v) for v in pred_list]) # 再割り当てされたデータ数
                        if any([i > threshold for i in counter]):  # 元のデータによって作られた空間にそのデータが割り振られた割合が90%を超えたら終了
                            flag = False
                        elif any([len(v) == 0 for v in pred_list]):  # ある集合のデータ数が0になっても終了
                            flag = False
                        else:
                            train_data = pred_list
                    all_class_subspaces.append(subspaces)
                print(f'subspaces: [{len(all_class_subspaces)}, {len(all_class_subspaces[0])}, {len(all_class_subspaces[0][0])}, {len(all_class_subspaces[0][0][0])}]')
                pred = accuracy(X_test, y_test, all_class_subspaces, dim)
                store_pred.append(pred)

            pred_mean = np.mean(store_pred)  # predの平均

            rec_rate.append(pred_mean*100)
            print('accuracy:', pred_mean)

        store_rec_rate.append(rec_rate)
        store_dimension.append(dimension)

    print(f"store_rec_rate={store_rec_rate}")
    print(f"store_dimension={store_dimension}")

    # 処理時間の計算
    elapsed_time = int(time.time() - start_time)
    # 処理時間の表示
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = elapsed_time % 3600 % 60
    print(f"elapsed time : {str(elapsed_hours).zfill(2) + ":"
                            + str(elapsed_minutes).zfill(2) + ":"
                            + str(elapsed_seconds).zfill(2)}")

    # 認識率と次元数のグラフ
    # 10種類の曲線スタイル
    line_formats = ['-', '--', ':', '-.', 'solid', 'dashed', 'dotted', 'dashdot', '-.', '--']
    for k_num in range(len(store_rec_rate)):
        plt.plot(store_dimension[k_num], store_rec_rate[k_num],
                 linestyle=line_formats[k_num], label=f'k={k_array[k_num]}')

    plt.title('k-SC')
    plt.xlabel('dim')
    plt.ylabel('rec_rate(%)')
    plt.legend()
    plt.grid(True)  # グリッド線を表示する
    plt.show()
    plt.savefig('k-SC')

    # 認識率と次元数のグラフ
    # 10種類の曲線スタイル
    line_formats = ['-', '--', ':', '-.', 'solid', 'dashed', 'dotted', 'dashdot', '-.', '--']
    for k_num in range(len(store_rec_rate)):
        plt.plot(store_dimension[k_num], store_rec_rate[k_num],
                 linestyle=line_formats[k_num], label=f'k={k_array[k_num]}')
    plt.ylim(90,)
    plt.title('k-SC')
    plt.xlabel('dim')
    plt.ylabel('rec_rate(%)')
    plt.legend()
    plt.grid(True)  # グリッド線を表示する
    plt.show()
    plt.savefig('k-SC')


if __name__ == '__main__':
    main()
