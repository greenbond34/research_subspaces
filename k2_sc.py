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
        def __init__(self, dim=784) -> None:
            self.dim = dim
        
        def like_kmeans():
            """k-meansのような処理"""
        
        def devide_into_k_pieces(self, Xtrain, ytrain, k):
            """各ラベルの学習サンプルをk個に分割する"""
            self.k = k
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
    # mnistの呼び出し(比較実験時にデータを同じにするため)
    X_train, X_test, y_train, y_test = load_mnist()

    k_sc = kSC(dim=dim)
    # 学習サンプルを0~9のラベル毎に分けて，分けたものをまとめたものの配列を返す
    num_Xlabel, num_ylabel = classify_by_label(X_train, y_train)

    # 処理時間の計算
    elapsed_time = int(time.time() - start_time)
    # 処理時間の表示
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = elapsed_time % 3600 % 60
    print(f"elapsed time : {str(elapsed_hours).zfill(2) + ":"
                            + str(elapsed_minutes).zfill(2) + ":"
                            + str(elapsed_seconds).zfill(2)}")
