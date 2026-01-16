import numpy as np
import pandas as pd
from scipy.stats import entropy, ttest_ind
from sklearn.feature_selection import SelectKBest, f_classif,mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#数据预处理
example = pd.read_table("partitioning_motif_f/3-AF1 binding site.txt")
example.set_index("gene_ID",inplace=True)
shape = example.shape #读取其中一个motif作为示例
non_partitioning_data = pd.read_csv('non_partitioning.txt', sep='\t')
non_partitioning_data.columns = ["gene_ID"] + non_partitioning_data.columns[1:].tolist()
non_partitioning_data.set_index('gene_ID', inplace=True)
partitioning_data = pd.read_csv('partitioning.txt', sep='\t')
partitioning_data.columns = ["gene_ID"] + partitioning_data.columns[1:].tolist()
partitioning_data.set_index('gene_ID', inplace=True)
df = pd.concat([non_partitioning_data,partitioning_data]) #合并所有motif信息成为一个数据框
non_partitioning_position = pd.read_csv("non_partitioning_position.txt",sep = "\t")
partitioning_position = pd.read_csv("partitioning_position.txt",sep = "\t")
non_partitioning_position.index = non_partitioning_position["gene_ID"]
partitioning_position.index = partitioning_position["gene_ID"]
partitioning_position.drop(["gene_ID"],inplace=True,axis=1)
non_partitioning_position.drop(["gene_ID"],inplace=True,axis=1)
partitioning_position = partitioning_position.replace([np.inf],2000)
non_partitioning_position = non_partitioning_position.replace([np.inf],2000)
features = non_partitioning_position.columns

#计算特征的交叉熵并筛选前m个特征的函数
def calculate_cross_entropies(non_data,par_data,m):
    """
       交叉熵计算

       Parameters:
       -----------
       non_data: pd_Dataframe，未分离组数据
       par_data: pd_Dataframe，分离组数据
       m: float

       Returns:
       --------
       top_ce_features：交叉熵前m的特征
       """
    features = non_data.columns
    df = pd.concat([non_data,par_data])
    X_log = np.log1p(df)
    df["label"] = [0]*len(non_data) + [1]*len(par_data)
    group2 = X_log[df["label"] == 0]
    group3 = X_log[df["label"] == 1]
    # 交叉熵
    cross_entropies = {}
    for f in features:
        # 获取联合值集合（对齐索引）
        all_values = pd.concat([group2[f], group3[f]]).unique()
        all_values = np.sort(all_values)  # 确保有序

        # 计算概率分布，填充缺失值为平滑值
        group2_counts = group2[f].value_counts().reindex(all_values, fill_value=0) + 1e-6
        group3_counts = group3[f].value_counts().reindex(all_values, fill_value=0) + 1e-6

        # 归一化
        group2_prob = group2_counts / group2_counts.sum()
        group3_prob = group3_counts / group3_counts.sum()

        # 交叉熵 H(P,Q) + H(Q,P) / 2
        h_pq = entropy(group2_prob, group3_prob)
        h_qp = entropy(group3_prob, group2_prob)
        avg_ce = (h_pq + h_qp) / 2
        cross_entropies[f] = avg_ce

    top_ce_features = sorted(cross_entropies, key=cross_entropies.get, reverse=True)[0:m]
    return top_ce_features

def calculate_vif(X, vif_threshold=5, max_iter=20):
    """
    迭代删除高VIF特征

    Parameters:
    -----------
    X : DataFrame, 特征矩阵
    vif_threshold : float, VIF阈值，默认10
    max_iter : int, 最大迭代次数

    Returns:
    --------
    X_selected : DataFrame, 筛选后的特征
    vif_results : DataFrame, VIF结果
    """
    X_temp = X.copy()
    dropped_features = []
    iteration = 0

    print(f"\n开始VIF检验，阈值={vif_threshold}")
    print("-" * 60)

    while iteration < max_iter:
        # 添加常数项用于VIF计算
        X_with_const = add_constant(X_temp)

        # 计算每个特征的VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_temp.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1)
                           for i in range(len(X_temp.columns))]

        vif_data = vif_data.sort_values("VIF", ascending=False)

        max_vif = vif_data["VIF"].iloc[0]
        max_vif_feature = vif_data["feature"].iloc[0]

        print(f"迭代 {iteration + 1}:")
        print(f"  最高VIF特征: {max_vif_feature} (VIF = {max_vif:.2f})")
        print(f"  剩余特征数: {len(X_temp.columns)}")

        if max_vif > vif_threshold:
            # 删除VIF最高的特征
            X_temp = X_temp.drop(columns=[max_vif_feature])
            dropped_features.append(max_vif_feature)
            print(f"  删除特征: {max_vif_feature}")
        else:
            print(f"  所有特征VIF <= {vif_threshold}，停止迭代")
            break

        iteration += 1

    print(f"\nVIF检验完成:")
    print(f"  最终特征数量: {len(X_temp.columns)}")
    print(f"  删除的特征数量: {len(dropped_features)}")
    if dropped_features:
        print(f"  删除的特征: {dropped_features}")

    return X_temp, vif_data


def calculate_feature_feature_mi_matrix(X, random_state=7):
    """
    计算特征间互信息矩阵的函数

    参数:
    ----------
    X : pandas DataFrame
        特征数据
    random_state : int, default=7
        随机种子

    返回:
    ----------
    mi_matrix : numpy array
        互信息矩阵，形状为 (n_features, n_features)
    feature_names : list
        特征名称列表
    """
    feature_names = X.columns.tolist()
    n_features = len(feature_names)

    # 初始化互信息矩阵
    mi_matrix = np.zeros((n_features, n_features))

    # 使用向量化方式提高计算效率（对于大数据集）
    # 但互信息计算需要逐个特征对进行
    for i in range(n_features):
        # 对角线设为1
        mi_matrix[i, i] = 1.0

        # 计算特征i与其他特征的互信息
        for j in range(i + 1, n_features):
            # 计算特征i和特征j之间的互信息
            mi_ij = mutual_info_regression(
                X.iloc[:, [i]],
                X.iloc[:, j],
                random_state=random_state
            )[0]
            mi_matrix[i, j] = mi_ij
            mi_matrix[j, i] = mi_ij

    return mi_matrix, feature_names


def get_high_mi_pairs(mi_matrix, feature_names, threshold=0.5):
    """
    获取互信息高于阈值的特征对

    参数:
    ----------
    mi_matrix : numpy array
        互信息矩阵
    feature_names : list
        特征名称列表
    threshold : float
        互信息阈值

    返回:
    ----------
    high_mi_pairs : list of tuples
        高互信息特征对列表，每个元组为 (特征1, 特征2, 互信息值)
    """
    n_features = len(feature_names)
    high_mi_pairs = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi_value = mi_matrix[i, j]
            if mi_value >= threshold:
                high_mi_pairs.append((feature_names[i], feature_names[j], mi_value))

    # 按互信息值降序排序
    high_mi_pairs.sort(key=lambda x: x[2], reverse=True)

    return high_mi_pairs

#位置特征提取函数
def imp_features(feat_list):
    imp_features = []
    for i,feat in enumerate(feat_list):
        if  i % 8 == 0:
            matches = feat.split("_max_")[0]
            imp_features.append(matches)
    return imp_features
position_feats = imp_features(features)



#count数信息和位置信息合并
non_ns = pd.DataFrame()
non_xm = pd.DataFrame()
par_ns = pd.DataFrame()
par_xm = pd.DataFrame()
non = pd.DataFrame()
par = pd.DataFrame()
for feat in position_feats:
    non_ns[feat + "_mean_Ns"] = non_partitioning_position[feat + "_mean_Ns"]
    non_xm[feat + "_mean_Xm"] = non_partitioning_position[feat + "_mean_Xm"]
    par_ns[feat + "_mean_Ns"] = partitioning_position[feat + "_mean_Ns"]
    par_xm[feat + "_mean_Xm"] = partitioning_position[feat + "_mean_Xm"]
    non[feat + "_mean_Ns"] = non_partitioning_position[feat + "_mean_Ns"]
    non[feat + "_mean_Xm"] = non_partitioning_position[feat + "_mean_Xm"]
    par[feat + "_mean_Ns"] = partitioning_position[feat + "_mean_Ns"]
    par[feat + "_mean_Xm"] = partitioning_position[feat + "_mean_Xm"]
non_interg = pd.concat([non_partitioning_data,non_partitioning_position],axis=1)
par_interg = pd.concat([partitioning_data,partitioning_position],axis=1)
interg = pd.concat([non_interg,par_interg],axis=0)
y = np.array([0]*len(non_interg.index) + [1]*len(par_interg.index))

#对所有motif的Ns和Xs做差形成新的数据
diff = pd.DataFrame()
for feat in position_feats:
    diff[feat + "count"] = np.abs(interg[feat + "_count_Ns"] - interg[feat + "_count_Xm"])
    diff[feat + "min"] = np.abs(interg[feat + "_min_Ns"] - interg[feat + "_min_Xm"])
    diff[feat + "mean"] = np.abs(interg[feat + "_mean_Ns"] - interg[feat + "_mean_Xm"])
    diff[feat + "median"] = np.abs(interg[feat + "_median_Ns"] - interg[feat + "_median_Xm"])
    diff[feat + "max"] = np.abs(interg[feat + "_max_Ns"] - interg[feat + "_max_Xm"])

#ANOVA筛选p值小于0.05的特征
diff_selector = SelectKBest(score_func=f_classif)
diff_new = diff_selector.fit_transform(diff, y)
selected_diff_indices = diff_selector.get_support(indices=True)
p_values = diff_selector.pvalues_
p_values = np.array(p_values)
significant_mask = p_values < 0.05
diff_feat = diff.columns[significant_mask].tolist()[0:30]

#对通过ANOVA的特征进行VIF检验
X_selected = diff[diff_feat]
X_vif_selected,vif_results = calculate_vif(X_selected)

#计算特征之间的互信息并筛选互信息小于0.5的特征
mi_tarix,feature_names = calculate_feature_feature_mi_matrix(X_selected)
high_mi_pairs = get_high_mi_pairs(mi_tarix,feature_names)

#计算交叉熵
top_ce_features = calculate_cross_entropies(non_interg,par_interg,50)


