import numpy as np
from copy import deepcopy

def simplex(A,b,c):
    # A：条件式の係数のみのmxn行列
    # b：不等式標準形の右辺の定数のみの1xm行列
    # c：目的関数の係数のみの1xm行列
    # m: 方程式の数
    # n: 変数の数
    m, n = A.shape

    # 辞書の初期化
    dic = np.zeros((m + 1, n + 1))
    for i in range(m):
        for j in range(n):
            dic[i + 1, j + 1] += A[i, j]
    for i in range(n):
        dic[0, i + 1] += c[i]
    for i in range(m):
        dic[i + 1, 0] += b[i]
    
    #print(dic)    
    
    # 非基底変数
    non_basis_variable = np.zeros(n)
    # 基底変数
    basis_variable = np.zeros(m)
    
    #各変数のindex、0-(n-1)とn-(m-1)
    non_basis_variable_index = np.arange(n)
    basis_variable_index = np.arange(n, m+n)
    
    # 2段階法が必要かチェック
    for x in basis_variable:
        if x < 0:
            # 本当は2段階法が必要
            print('この制約条件には2段階法を適用する必要があります.')
            return
    
    # 原点が許容解であるか確認
    for i in range(n):
        #　初期の非基底変数を全て0にする
        non_basis_variable[i] = 0
    
    # 非基底変数が原点のときの基底変数を代入
    basis_variable = np.dot(dic[1:, 1:], non_basis_variable) + dic[1:, 0]
    
    # 最適解
    opt_sol = loop(dic, m, n, non_basis_variable, basis_variable, basis_variable_index, non_basis_variable_index)
    # 最適解が得られるときの決定変数値
    val_sol = [0 for _ in range(n)]
    
    all_val = np.concatenate([non_basis_variable, basis_variable])
    all_val_index = np.concatenate([non_basis_variable_index, basis_variable_index])
    
    #愚直に代入
    for j, x in zip(all_val_index, all_val):
        if j < n:
            val_sol[j] = x

    return opt_sol, np.array(val_sol)


def loop(dic, m, n, non_basis_variable, basis_variable, basis_variable_index, non_basis_variable_index):
    while True:
        # 非基底変数を決定
        j_pivot = -1
        for j, tmp in enumerate(dic[0, 1:]):
            if tmp < 0:
                # 負の係数が見つかったらそこを非基底変数にする
                j_pivot = j + 1
                break
        if j_pivot == -1:
            # 負の係数が見つからなかったら終了
            return np.dot(dic[0, 1:], non_basis_variable) + dic[0, 0]

        # 基底変数を決定
        base_candidate = {}
        # 制約条件を順に調べる
        for i in range(1, m+1):
            # 非規定変数の値が負の時に
            if dic[i, j_pivot] < 0:
                base_candidate[i] = dic[i, 0] / abs(dic[i, j_pivot])
                    
        #どこまでなら変数値を上げられるかを決定
        d_min = min(base_candidate.values())
        i_min_list = [kv[0] for kv in base_candidate.items() if kv[1] == d_min]
        
        # 基底の候補から添え字の最小となる基底を選択
        i_pivot = m + n
        index_pivot = m + n
        for i_min in i_min_list:
            if index_pivot >= basis_variable_index[i_min - 1]:
                i_pivot = i_min
                index_pivot = basis_variable_index[i_min - 1]

        # 選択された非基底変数を最大限まで増加させる
        non_basis_variable[j_pivot - 1] = d_min
        basis_variable[i_pivot - 1] = 0.0

        # 変数の更新
        basis_variable[i_pivot - 1], non_basis_variable[j_pivot - 1] = non_basis_variable[j_pivot - 1], basis_variable[i_pivot - 1]
        basis_variable_index[i_pivot - 1], non_basis_variable_index[j_pivot - 1] = non_basis_variable_index[j_pivot - 1], basis_variable_index[i_pivot - 1]

        dic_prev = dic.copy()
        for i in range(m + 1):
            if i == i_pivot:
                c = -1.0 / dic_prev[i_pivot, j_pivot]
                dic[i, :] = dic_prev[i, :] * c
                dic[i_pivot, j_pivot] = -c
            else:
                c = dic_prev[i, j_pivot] / dic_prev[i_pivot, j_pivot]
                dic[i, :] = dic_prev[i, :] - dic_prev[i_pivot, :] * c
                dic[i, j_pivot] = c

def show_result(sol, dec_val_dic):
    n = dec_val_dic.shape[0]
    print("最適解：",sol)
    for i in range(n):
        print(f"変数{i+1}：{dec_val_dic[i]}")

# 実行する場所
A = np.array([[1,2],[1,1],[3,1]])
b = np.array([10,6,12])
c = np.array([2,1])
sol, dec_val_dic = simplex(A, b, c)
show_result(sol, dec_val_dic)