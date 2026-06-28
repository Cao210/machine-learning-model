import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义评估指标函数
def mse(output, label):
    return mean_squared_error(label, output)

def rmse(output, label):
    return np.sqrt(np.mean((output - label) **2))

def mae(output, label):
    return mean_absolute_error(label, output)

def r2(output, label):
    return r2_score(label, output)

def mape(output, label):
    return np.mean(np.abs((label - output) / (label + 1e-8))) * 100  # 加小值避免除零

# 读取数据
file_path = r"E:\论文报告\杨傲数据\杨傲.xlsx"  # 保持原数据路径
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 选取特征和目标列（保持原特征目标列）
features = ['龄期', '取代率']
targets = ['抗压', '抗折']

# 检查列名是否存在
missing_cols = [col for col in features + targets if col not in df.columns]
if missing_cols:
    raise KeyError(f"以下列名不存在：{missing_cols}")

data = df[features + targets].dropna()

# 划分特征矩阵和目标矩阵
X = data[features].apply(pd.to_numeric, errors='coerce')
y = data[targets].apply(pd.to_numeric, errors='coerce')

# 处理无效值并对齐长度
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.replace([np.inf, -np.inf], np.nan).dropna()
min_len = min(len(X), len(y))
X, y = X.iloc[:min_len], y.iloc[:min_len]

# 划分训练集和测试集（与随机森林一致：85:15）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)


# 构建线性回归模型管道（保留标准化，线性回归需要特征标准化）
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 特征标准化
    ('linear', LinearRegression())  # 线性回归模型
])

# 定义超参数搜索空间
param_dist = {
    'linear__fit_intercept': [True, False]  # 是否计算截距
}

# 初始化随机搜索与模型训练（与随机森林参数逻辑一致）
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,  # 与随机森林一致的搜索迭代次数
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    return_train_score=True  # 新增：返回训练集分数
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("最佳超参数:", random_search.best_params_)

# 预测结果（先测试集后训练集，与随机森林顺序一致）
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# 打印预测值和对应的真实值（与随机森林枚举方式一致）
for target_index, target in enumerate(targets):
    print(f"\n{target} 的测试集预测值和真实值：")
    print(pd.DataFrame({
        '真实值': y_test.iloc[:, target_index],
        '预测值': y_pred_test[:, target_index]
    }))
    print(f"\n{target} 的训练集预测值和真实值：")
    print(pd.DataFrame({
        '真实值': y_train.iloc[:, target_index],
        '预测值': y_pred_train[:, target_index]
    }))

# 计算并展示评估指标（与随机森林存储结构和顺序一致）
results_test = []
results_train = []
for i, target in enumerate(targets):
    # 测试集评估指标
    test_mse = mse(y_test.iloc[:, i], y_pred_test[:, i])
    test_rmse = rmse(y_test.iloc[:, i], y_pred_test[:, i])
    test_mae = mae(y_test.iloc[:, i], y_pred_test[:, i])
    test_r2 = r2(y_test.iloc[:, i], y_pred_test[:, i])
    test_mape = mape(y_test.iloc[:, i], y_pred_test[:, i])
    results_test.append({
        '评估指标': ['均方误差 (MSE)', '均方根误差 (RMSE)', '平均绝对误差 (MAE)', '决定系数 (R²)', '平均绝对百分比误差 (MAPE)'],
        '测试集得分': [test_mse, test_rmse, test_mae, test_r2, test_mape],
        '目标变量': target
    })

    # 训练集评估指标
    train_mse = mse(y_train.iloc[:, i], y_pred_train[:, i])
    train_rmse = rmse(y_train.iloc[:, i], y_pred_train[:, i])
    train_mae = mae(y_train.iloc[:, i], y_pred_train[:, i])
    train_r2 = r2(y_train.iloc[:, i], y_pred_train[:, i])
    train_mape = mape(y_train.iloc[:, i], y_pred_train[:, i])
    results_train.append({
        '评估指标': ['均方误差 (MSE)', '均方根误差 (RMSE)', '平均绝对误差 (MAE)', '决定系数 (R²)', '平均绝对百分比误差 (MAPE)'],
        '训练集得分': [train_mse, train_rmse, train_mae, train_r2, train_mape],
        '目标变量': target
    })

# 输出结果（先测试集后训练集，与随机森林顺序一致）
print("\n测试集评估结果：")
for result in results_test:
    df_result = pd.DataFrame(result)
    print(f"\n{result['目标变量']} 评估结果：")
    print(df_result)

print("\n训练集评估结果：")
for result in results_train:
    df_result = pd.DataFrame(result)
    print(f"\n{result['目标变量']} 评估结果：")
    print(df_result)
