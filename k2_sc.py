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
    X = data[np.where(target == str(c))[0]]      # class c のデータを取得
    # N, D = X.shape                              # データ数, データ次元数
    y = target[np.where(target == str(c))[0]]

    num_Xlabel.append(X)
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
            print(f"norm_list={norm_list}")
            # 距離が最も近いクラスを予測クラスとする
            pred_list[i] = np.argmax(norm_list)
        print(f"pred_list={pred_list}")
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
        for row_target in tqdm(range(self.k)):
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


def main():
    """main"""
    dim = 40
    k = 3
    # mnistの呼び出し(比較実験時にデータを同じにするため)
    X_train, X_test, y_train, y_test = load_mnist()

    k_sc = kSC(dim=dim)
    # 学習サンプルを0~9のラベル毎に分けて，分けたものをまとめたものの配列を返す
    num_Xlabel, num_ylabel = classify_by_label(X_train, y_train)
    
    for i in range(10):
        k_sc.fit(num_Xlabel[i], num_ylabel[i], k)

    # 処理時間の計算
    elapsed_time = int(time.time() - start_time)
    # 処理時間の表示
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = elapsed_time % 3600 % 60
    print(f"elapsed time : {str(elapsed_hours).zfill(2) + ":"
                            + str(elapsed_minutes).zfill(2) + ":"
                            + str(elapsed_seconds).zfill(2)}")
