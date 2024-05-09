"""plot {Recognition_rate * dimension} (subspace classifier)
elapsed time : 01:23:04"""
import time
import ssl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


start_time = time.time()
ssl._create_default_https_context = ssl._create_unverified_context


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


# 部分空間法
class Calfic():
    """部分空間法"""

    def __init__(self, dim=784) -> None:
        self.dim = dim
        # self.subspaces = []
        # self.set_subspaces = []

    def fit(self, X_train, y_train, k):
        """部分空間を作成しているだけ"""
        self.k = k
        self.set_subspaces = self.k_subspaces(X_train, y_train)
        # self.subspaces = self.subspace(X_train, y_train, k)
        # self.set_subspaces.append(self.subspaces)

    # 部分空間法で予測
    def predict(self, data_test) -> np.ndarray:
        """部分空間法で予測する"""
        num_data = len(data_test)
        num_classes = len(self.set_subspaces[0])
        print('Start Prediction')

        pred_list = np.empty(num_data, dtype=int)
        norm_list = np.empty(num_classes)

        for i in tqdm(range(num_data)):
            for j in range(num_classes):
                norm_k = 0
                for subspace in self.set_subspaces:
                    # 各クラスの空間との距離を計算（ユークリッド）
                    norm = sum(np.dot(data_test[i, :], subspace[j, :, k]) ** 2 for k in range(self.dim))
                    if norm > norm_k:
                        norm_k = norm
                norm_list[j] = norm_k
            # 距離が最も近いクラスを予測クラスとする
            pred_list[i] = np.argmax(norm_list)
        print('Prediction Completed')
        return pred_list

    # 部分空間法の精度を算出
    def accuracy(self, data_test, target_test) -> float:
        """部分空間法の精度を算出"""
        num_data = len(data_test)
        pred_list = self.predict(data_test)
        count = 0
        for i in range(num_data):
            if pred_list[i] == int(target_test[i]):
                count += 1
        acc = count / num_data
        print('accuracy is: ', acc)
        return acc

    def k_subspaces(self, data, target) -> np.ndarray:
        """k組の部分空間を作成

        Args:
            data (_type_): 学習データ()
            target (_type_): 正解データ

        Returns:
            np.ndarray: k組*クラス数の部分空間
        """
        print('k:', self.k)
        print('Start creating Subspaces')
        # print(f"target.shape={target.shape}")
        test_list = []
        for row_target in range(self.k):
            print(f'{row_target+1}組目')
            test_list.append(self.subspace(data[row_target], target[row_target]))
        return np.array(test_list)

    # 部分空間を作成(次元数はdimで指定)
    def subspace(self, data, target) -> np.ndarray:
        """部分空間を作成(次元数はdimで指定)"""
        lambdas = []
        class_eigenvectors = []
        print('Start creating Subspaces')
        for c in tqdm(np.unique(target)):
            X = data[np.where(target == str(c))[0]]      # class c のデータを取得
            N, D = X.shape                              # データ数, データ次元数
            for i in range(D):
                if len(np.unique(X[:, i])) == 1:
                    X[np.random.randint(0, N), i] = 1     # おまじない（分散が0になるのを防ぐ）
            correction = np.corrcoef(X.T)               # 自己相関行列を求める
            lam, vec_e = np.linalg.eigh(correction)     # 固有値と固有ベクトルを求める
            vec_e = vec_e[:, np.argsort(lam)[::-1]]     # 固有値の降順に固有ベクトルを並べ替える
            class_eigenvectors.append(vec_e)            # 固有ベクトルを格納
            lambdas.append(lam)
        class_eigenvectors = np.array(class_eigenvectors)
        class_subspaces = []
        for eigenvectors in class_eigenvectors:
            subspace = eigenvectors[:, :self.dim]
            class_subspaces.append(subspace)
        class_subspaces = np.array(class_subspaces)
        print('Subspaces Created')
        return class_subspaces


def split_train_data(X_train, y_train, k):
    """学習サンプルをｋ組に分ける"""
    # train_dataをシャッフルする
    np.random.seed(10)
    np.random.shuffle(X_train)
    np.random.seed(10)
    np.random.shuffle(y_train)
    # 分割する数
    num_sets = k

    # train_dataをnum_of_splitsに分割する
    split_ind = np.array_split(np.arange(len(X_train)), num_sets)

    # 分割したインデックスを使用してtrain_dataを分割する
    train_array_splits = [X_train[idx] for idx in split_ind]
    target_array_splits = [y_train[idx] for idx in split_ind]

    return train_array_splits, target_array_splits


def main():
    """main"""
    # mnistをローカルに保存できたからコメントアウト
    # set_mnist()                                         # 初回のみ実行(mnistをローカルに作成)

    times = 10  # 最大の次元数
    bunkatu = 2  # 分割数

    times = times + 1
    bunkatu = bunkatu + 1

    # plotのために学習サンプルをk組に分けた時のrec_rateとdimensionを保管する
    store_rec_rate = []
    store_dimension = []

    max_rec_rate = []
    max_dim = []
    k_array = []

    num_sets = bunkatu
    # 学習サンプルをkに分割
    k = num_sets
    k_array.append(k)
    # k個の学習サンプルを入れる箱
    split_Xtrain = []
    split_ytrain = []
    # 認識率と次元の数値を次元ごとに保存する
    rec_rate = []
    dimension = []

    for i in range(1, times):
        dim = i                                           # 部分空間の次元数を設定(ここを変えて精度の変化を見る)
        dimension.append(dim)
        print(f"k={k} {dim}次元")

        X_train, X_test, y_train, y_test = load_mnist()     # mnistの呼び出し(比較実験時にデータを同じにするため)
        clf = Calfic(dim=dim)                               # 部分空間法の呼び出し: 空間の次元数はdimで指定
        split_Xtrain, split_ytrain = split_train_data(X_train, y_train, k)

        clf.fit(split_Xtrain, split_ytrain, k)  # 部分空間法の学習

        # pred = clf.predict(X_test)                        # 部分空間法でテストデータを予測(予測データ取り出したい時用)
        acc = clf.accuracy(X_test, y_test)                  # 部分空間法の識別率を計算
        rec_rate.append(acc*100)
        print('accuracy:', acc)

        store_rec_rate.append(rec_rate)
        store_dimension.append(dimension)

    print(f"store_rec_rate={store_rec_rate}")
    print(f"store_dimension={store_dimension}")

    # k毎のピーク性能を比較するため，k毎のピークを集める
    for i in range(len(store_rec_rate)):
        is_peak = 0
        is_dim = 0
        for j in range(len(store_rec_rate[i])):
            if is_peak < store_rec_rate[i][j]:
                is_peak = store_rec_rate[i][j]
                is_dim = j
        max_rec_rate.append(is_peak)
        max_dim.append(is_dim)

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
                 linestyle=line_formats[k_num], label=f'k={k_num+1}')

    plt.title('k-SC')
    plt.xlabel('dim')
    plt.ylabel('rec_rate(%)')
    plt.legend()
    plt.grid(True)  # グリッド線を表示する
    plt.show()
    plt.savefig('k-SC')

    # 棒グラフ
    for x_coo, rate in enumerate(max_rec_rate):
        plt.text(x_coo+1, rate // 3, f"(dim={max_dim[x_coo]})  {str(rate)}",
                 ha='center', rotation='vertical')
    plt.bar(k_array, max_rec_rate)
    plt.title('peak rec_rate per k')
    plt.xlabel('k')
    plt.ylabel('peak rec_rate(%)')
    plt.grid(True)  # グリッド線を表示する
    plt.show()


if __name__ == '__main__':
    main()
