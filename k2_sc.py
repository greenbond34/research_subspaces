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
    """
    学習サンプルを0~9のラベル毎に分割,分割したものを配列にまとめて返す
    data : X_train（mainのところで)
    target : y_train（mainのところで)
    """
    num_Xlabel = []
    num_ylabel = []
    for c in np.unique(target):
        x = data[np.where(target == str(c))[0]]      # class c のデータを取得
        # N, D = X.shape                              # データ数, データ次元数
        y = target[np.where(target == str(c))[0]]

    num_Xlabel.append(x)
    num_ylabel.append(y)

    return num_Xlabel, num_ylabel


class kSC():
    """k-部分空間法"""
    def __init__(self, dim=784):
        self.dim = dim

    def fit(self, X_train, y_train, k):
        self.k = k
        # 学習サンプルをｋ個に分割
        set_Xcluster, set_ycluster = self.devide_into_k_pieces(X_train, y_train)
        # k組の部分空間を作成
        self.set_subspaces = self.k_subspaces(set_Xcluster, set_ycluster)

    def like_kmeans(self, X_train, y_train):
        """ｋ個の各部分空間に含まれているサンプルで，それぞれの部分空間を作る．
        学習サンプルを写像
        それを繰り返す
        引数：一つのラベル分の学習サンプル
        """
        self.X_train = X_train
        self.y_train = y_train

        # 一つのラベル分の学習サンプルを写像し，予測結果を受け取る
        pred_list = self.predict(self.X_train)

        # 繰り返しは，収束条件をつけたループ文にする
        for _ in range(5):
            # X_train,y_trainの中身を所属のクラス毎に分ける
            cluster_X, cluster_y = self.separate_contents(pred_list)
            # ｋ個の各部分空間に含まれているサンプルで，それぞれの部分空間を作る
            self.set_subspaces = self.k_subspaces(cluster_X, cluster_y)
            # 一つのラベル分の学習サンプルを写像し，予測結果を受け取る
            pred_list = self.predict(self.X_train)

        return self.set_subspaces

    def convergence_condi(self, n_train, n_1_train):
        """
        収束条件
        n回目のαクラスに所属している学習サンプル(n_train)が
        n-1回目のαクラスに所属していた学習サンプル(n_1_train)の
        x%同じなら収束したとみなす"""
        x = 100  # 何%にするか，重ならないから100%
        count_true = 0
        for _ in range(self.k):
            true_rate = 0
            exit_true = 0
            for i in enumerate(n_train):
                if n_train[i] in n_1_train:
                    exit_true += 1
            true_rate = (count_true / len(n_train)) * 100
            if true_rate >= x:
                count_true += 1
        if count_true == self.k:
            return False
        return True

    def separate_contents(self, pred_list):
        """X_train,y_trainの中身を所属のクラス毎に分ける"""
        # pred_list内のインデックスをクラス毎にまとめる
        k_indexes = []
        for labels in np.unique(pred_list):
            index = np.where(labels == pred_list)
            k_indexes.append(index)
        # X_train,y_trainの特徴ベクトルをクラス毎にまとめる
        cluster_X = []
        cluster_y = []
        for i in range(len(k_indexes)):
            cluster_Xlabels = []
            cluster_ylabels = []
            for j in k_indexes[i]:
                cluster_Xlabels.append(self.X_train[j])
                cluster_ylabels.append(self.y_train[j])
            cluster_X.append(cluster_Xlabels)
            cluster_y.append(cluster_ylabels)

        cluster_X = np.array(cluster_X)
        cluster_y = np.array(cluster_y)
        return cluster_X, cluster_y

        # # k個の部分空間を作成
        # self.set_subspaces = self.k_subspaces(cluster_X, cluster_y)
        # # 一つのラベル分の学習サンプルを写像し，予測結果を受け取る
        # pred_list = self.predict(self.X_train)

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

    # 部分空間法で予測（最終評価用）
    def predict_for_acc(self, data_test, all_subspaces) -> np.ndarray:
        """部分空間法で予測する"""
        num_data = len(data_test)
        num_classes = len(self.set_subspaces[0])*len(all_subspaces)
        print('Start Prediction')

        pred_list = np.empty(num_data, dtype=int)
        norm_list = np.empty(num_classes)

        for i in tqdm(range(num_data)):
            for j in range(num_classes):
                norm_k = 0
                for index in range(len(all_subspaces)):
                    for subspace in all_subspaces[index]:
                        # 各クラスの空間との距離を計算（ユークリッド）
                        norm = sum(np.dot(data_test[i, :], subspace[j, :, k]) ** 2 for k in range(self.dim))
                        if norm > norm_k:
                            norm_k = norm
                norm_list[j] = norm_k
            # 距離が最も近いクラスを予測クラスとする
            pred_list[i] = np.argmax(norm_list)
        print('Prediction Completed')
        return pred_list

    def devide_into_k_pieces(self, X_train, y_train):
        """各ラベルの学習サンプルをk個に分割する"""
        np.random.seed(10)
        np.random.shuffle(X_train)
        np.random.seed(10)
        np.random.shuffle(y_train)
        # train_dataをnum_of_splitsに分割する
        split_ind = np.array_split(np.arange(len(X_train)), self.k)
        # 分割したインデックスを使用してtrain_dataを分割する
        train_array_splits = [X_train[idx] for idx in split_ind]
        target_array_splits = [y_train[idx] for idx in split_ind]
        return train_array_splits, target_array_splits

        # 部分空間法の精度を算出
    def accuracy(self, data, target, all_subspaces) -> float:
        """精度の算出"""
        num_data = len(data)
        pred_list = self.predict_for_acc(data, all_subspaces)
        count = 0
        for i in tqdm(range(num_data)):
            if pred_list[i] == int(target[i]):
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
        print(f"type(target)={type(target)}")
        for c in tqdm(np.unique(target)):
            print(f"type(c)={type(c)}")
            X = data[np.where(target == c)[0]]      # class c のデータを取得
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


def main():
    """main"""
    dim = 40
    k = 3
    all_subspaces = []
    # mnistの呼び出し(比較実験時にデータを同じにするため)
    X_train, X_test, y_train, y_test = load_mnist()

    k_sc = kSC(dim=dim)
    # 学習サンプルを0~9のラベル毎に分けて，分けたものをまとめて配列で返す
    num_Xlabel, num_ylabel = classify_by_label(X_train, y_train)

    for i in range(10):
        k_sc.fit(num_Xlabel[i], num_ylabel[i], k)
        set_subspaces = k_sc.like_kmeans(num_Xlabel[i], num_ylabel[i])
        all_subspaces.append(set_subspaces)

    acc = k_sc.accuracy(X_test, y_test, all_subspaces)                  # 部分空間法の識別率を計算
    print('accuracy:', acc)

    # 処理時間の計算
    elapsed_time = int(time.time() - start_time)
    # 処理時間の表示
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = elapsed_time % 3600 % 60
    print(f"elapsed time : {str(elapsed_hours).zfill(2) + ":"
                            + str(elapsed_minutes).zfill(2) + ":"
                            + str(elapsed_seconds).zfill(2)}")


if __name__ == '__main__':
    main()
