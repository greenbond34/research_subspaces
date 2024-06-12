"""k-subspace classifier"""
import time
import ssl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import ray
from sklearn.cluster import KMeans
# 自作モジュール
from modules.mnist import load_mnist

# 並列処理を使うか選択
MULTIPROCESS = True

ssl._create_default_https_context = ssl._create_unverified_context


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"function[{func.__name__}] 経過時間: ", end_time - start_time, "seconds")
        return result
    return wrapper


def make_subspace(data, dim) -> np.ndarray:
    """データから部分空間を作成 """
    N, D = len(data), len(data[0])               # データ数, データ次元数
    for i in range(D):
        if len(np.unique(data[:, i])) == 1:
            data[np.random.randint(0, N), i] = 1   # おまじない（分散が0になるのを防ぐ）
    correction = np.corrcoef(data.T)             # 自己相関行列を求める
    lam, vec_e = np.linalg.eigh(correction)      # 固有値と固有ベクトルを求める
    vec_e = vec_e[:, np.argsort(lam)[::-1]]      # 固有値の降順に固有ベクトルを並べ替える
    subspace = vec_e[:, :dim]
    return subspace


def clustering_by_subspace(test_data, subspaces, k, dim) -> np.ndarray:
    """k個の部分空間に学習データを写像し、近い空間ごとにクラスタリング"""
    counter = np.zeros(k)
    norm_list = np.empty(k)
    pred_list = [[] for _ in range(k)]
    for v, data in enumerate(test_data):
        for i in range(len(data)):
            for j in range(k):
                norm_list[j] = sum(np.dot(data[i, :], subspaces[j, :, v]) ** 2 for v in range(dim))
            pred_list[np.argmax(norm_list)].append(data[i])
            if v == np.argmax(norm_list):
                counter[v] += 1
        counter[v] /= len(data)
        print(f'counter{v}: {counter[v]}')
    for i in range(len(pred_list)):
        pred_list[i] = np.array(pred_list[i])
    return pred_list, counter


# 部分空間法で予測
@timer
def predict(data_test, subspaces, dim) -> np.ndarray:
    """部分空間法で予測する"""
    num_data, num_classes = len(data_test), len(subspaces)
    print('Start Prediction')
    if MULTIPROCESS:
        # 並列処理
        multiyou = [(data_test, subspaces[i], dim) for i in range(num_classes)]
        norm_list = ray.get([wrapper_calc_norm.remote(i) for i in multiyou])
    else:
        norm_list = [calcurate_norm(data_test, subspaces[i], dim) for i in range(num_classes)]
    # 全てのクラスに対する計算が終了した後に、予測クラスを決定
    pred_list = np.empty(num_data, dtype=int)
    for i in range(num_data):
        pred_list[i] = np.argmax([norm_list[j][i] for j in range(num_classes)])
    print('Prediction Completed')
    return pred_list


@ray.remote
def wrapper_calc_norm(args):
    return calcurate_norm(*args)


def calcurate_norm(data_test, subspace, dim):
    norm_k_list = np.zeros(len(data_test))
    for k in range(len(subspace)):
        for i in range(len(data_test)):
            # 各クラスの空間との距離を計算（ユークリッド）
            norm = sum(np.dot(data_test[i, :], subspace[k, :, v]) ** 2 for v in range(dim))
            if norm > norm_k_list[i]:
                norm_k_list[i] = norm
    return norm_k_list


# 部分空間法の精度を算出
def accuracy(data_test, target_test, subspaces, dim) -> float:
    """部分空間法の精度を算出"""
    num_data = len(data_test)
    pred_list = predict(data_test, subspaces, dim)
    count = 0
    for i in tqdm(range(num_data)):
        if pred_list[i] == int(target_test[i]):
            count += 1
    acc = count / num_data
    print('accuracy is: ', acc)
    return acc


@timer
def make_allclass_subspaces(X_train, y_train, k=5, dim=10, threshold=0.90):
    bar = [X_train[np.where(y_train == str(label))[0]] for label in np.unique(y_train)]
    if MULTIPROCESS:
        # 並列処理
        multiyou = [(bar[i], i, k, dim, threshold) for i in range(len(bar))]
        all_class_subspaces = ray.get([wrapper_sub.remote(i) for i in multiyou])
    else:
        # 非並列処理
        all_class_subspaces = [make_subspaces_for_a_number(bar[i], i, k, dim, threshold)for i in range(len(bar))]
    return all_class_subspaces


@ray.remote
def wrapper_sub(args):
    return make_subspaces_for_a_number(*args)


def make_subspaces_for_a_number(X_train, y, k, dim, threshold, clustering='kmeans'):
    def split_train_data(X_train, k):
        """学習サンプルをｋ組に分ける"""
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        return np.array_split(X_train, k)

    def kmeans(X_train, k):
        cluster = KMeans(n_clusters=k)
        cluster.fit(X_train)
        labels = cluster.labels_
        split_X_train = [X_train[np.where(labels == i)[0]] for i in range(k)]
        return split_X_train

    print(f'class: {y}')
    # while前の定義
    # 最初の分割方法の選択
    if clustering == 'random':
        # ランダムに分割
        train_data = split_train_data(X_train, k=k)
    elif clustering == 'kmeans':
        train_data = kmeans(X_train, k=k)
        # k-means法で分割
    flag = True
    loop_counter = 0
    while flag:
        loop_counter += 1
        print(f'loop: {loop_counter}')
        subspaces = np.empty((k, len(X_train[0]), dim))  # (k, data_dim, reduced_dim)
        for i, data in enumerate(train_data):
            subspaces[i] = make_subspace(data, dim=dim)                   # k組目のデータから部分空間を作成
        pred_list, counter = clustering_by_subspace(train_data, subspaces, k, dim)
        # While終了判定
        if any([i > threshold for i in counter]):  # 元のデータによって作られた空間にそのデータが割り振られた割合が90%を超えたら終了
            flag = False
        elif any([len(v) == 0 for v in pred_list]):  # ある集合のデータ数が0になっても終了
            flag = False
        else:
            train_data = pred_list
    return subspaces


@timer
def main():
    # 並列処理をする場合はrayを起動
    # MULTIPROCESSはimport文下に記載
    if MULTIPROCESS:
        ray.init()

    data, target = fetch_openml('mnist_784', version=1, return_X_y=True)
    data = np.array(data)
    target = np.array(target)

    max_k = 5
    max_dim = 60
    threshold = 0.90

    # plotのために学習サンプルをk組に分けた時のrec_rateとdimensionを保管する
    store_rec_rate = []
    store_dimension = []
    k_array = []

    for k in range(1, max_k+1):
        k_array.append(k)
        # 認識率と次元の数値を次元ごとに保存する
        rec_rate = []
        dimension = []

        for dim in range(1, max_dim+1):
            dimension.append(dim)
            print(f"k={k} {dim}次元")

            for divi in range(5):
                store_pred = []

                X_train, X_test, y_train, y_test = load_mnist()
                X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.16, random_state=divi)
                all_class_subspaces = make_allclass_subspaces(X_train, y_train, k, dim, threshold)
                acc = accuracy(X_test, y_test, all_class_subspaces, dim)
                store_pred.append(acc)

            pred_mean = np.mean(store_pred)  # predの平均

            rec_rate.append(pred_mean*100)
            print('accuracy:', pred_mean)

        store_rec_rate.append(rec_rate)
        store_dimension.append(dimension)

    print(f"store_rec_rate={store_rec_rate}")
    print(f"store_dimension={store_dimension}")

    # 認識率と次元数のグラフ
    # 10種類の曲線スタイル
    line_formats = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), 'dashed', 'dotted', 'dashdot', '-.', '--']
    for k_num in range(len(store_rec_rate)):
        plt.plot(store_dimension[k_num], store_rec_rate[k_num],
                 linestyle=line_formats[k_num], label=f'k={k_array[k_num]}')
    plt.ylim(90,)
    plt.title('k-SC_1')
    plt.xlabel('dim')
    plt.ylabel('rec_rate(%)')
    plt.legend()
    plt.grid(True)  # グリッド線を表示する
    plt.savefig('k-SC__upto60dim_min90%')

    # 認識率と次元数のグラフ
    # 10種類の曲線スタイル
    line_formats = ['-', '--', ':', '-.', 'solid', 'dashed', 'dotted', 'dashdot', '-.', '--']
    for k_num in range(len(store_rec_rate)):
        plt.plot(store_dimension[k_num], store_rec_rate[k_num],
                 linestyle=line_formats[k_num], label=f'k={k_array[k_num]}')

    plt.title('k-SC_2')
    plt.xlabel('dim')
    plt.ylabel('rec_rate(%)')
    plt.legend()
    plt.grid(True)  # グリッド線を表示する
    plt.savefig('k-SC__upto60dim')


if __name__ == '__main__':
    main()
