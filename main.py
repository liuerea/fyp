from sklearn import svm
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

'''
def plot_confusion_matrix(cm, labels_name, title):
    # cm = cm / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title, fontsize = 50)  # 图像标题
    plt.colorbar() #添加自定义颜色条
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, fontsize=30, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=30, rotation=0)  # 将标签印在y轴坐标上
    plt.ylabel('True label', fontsize=40) #纵轴标签
    plt.xlabel('Predicted label', fontsize=40) # 横轴标签
    #plt.figure(figsize=(2, 2), dpi=300)
'''

# 加载数据
path = 'wine.data'
names = ['ID number', 'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
         'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

data = pd.read_csv(path, header=None, names=names)
data.to_csv( 'wine.csv' )

# 显示数据
pd.set_option('display.max_columns', None)
# print(data.columns)
# print(data.head())
# print(data.describe())

# 诊断结果B良性：0，M恶性：1
# print(type(data))

# print(data)

# 特征字段分组，mean、se、worst
feature_mean = list(data.columns[1:14])


# 数据清洗，去掉ID
# data.drop(columns=['ID number'],axis=1,inplace=True)

# 诊断结果可视化
# sns.countplot(data['Diagnosis'],label='Count')
corr = data[feature_mean].corr()
plt.figure(figsize=(17, 17))
sns.heatmap(corr, annot=True)
# plt.show()

# 缩减属性（去除高度相关的，减小计算量）
feature_remain = ['Alcohol' , 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Flavanoids',
         'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'Proline']

# 抽取数据，6：2：2
train, test1 = train_test_split(data, test_size=0.5, random_state=111)
val, test = train_test_split(test1, test_size=0.4, random_state=121)
# print(len(train), len(val), len(test))

train_x = train[feature_remain]
train_y = train['ID number']
val_x = val[feature_remain]
val_y = val['ID number']
test_x = test[feature_remain]
test_y = test['ID number']

# 对数据进行z-score归一化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.fit_transform(val_x)
test_x = ss.fit_transform(test_x)

# 创建SVM分类器
print("-------SVC_调参前---------")
clf = svm.SVC()
clf.fit(train_x, train_y)
svc_predictions = clf.predict(val_x)
print(confusion_matrix(val_y, svc_predictions))
print(classification_report(val_y, svc_predictions))


Kernel = ["linear" , "poly", "rbf", "sigmoid"]
print("-------SVC_选择核函数---------")

for kernel in Kernel:
    svc_model= svm.SVC(kernel=kernel).fit(train_x, train_y)
    svc_validation = svc_model.predict(val_x)
    print(kernel, 'val acc: ', accuracy_score(svc_validation, val_y))


# 选择rbf高斯通核函数
rbf_model = svm.SVC(kernel="rbf")
rbf_model.fit(train_x, train_y)

# 验证集
print('SVC_rbf_验证集')
rbf_predictions = rbf_model.predict(val_x)
print(confusion_matrix(val_y, rbf_predictions))
print(classification_report(val_y, rbf_predictions))

print('SVC_rbf_参数训练')
param_grid = {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,100,1000],
              'gamma':[10,1,0.5,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.001,0.0001]}
# param_grid = {'C':[0.3] , 'gamma' : [0.08]}
kflod = StratifiedKFold(n_splits=10, shuffle = True , random_state=7)
svc_rbf_grid_model = GridSearchCV(svm.SVC(), param_grid , cv = kflod, verbose=0)
svc_rbf_grid_model.fit(train_x, train_y)
# print(grid_model)
print(svc_rbf_grid_model.best_params_)

print('SVC_rbf_调参后最终验证集')
svc_rbf_grid_predictions_val = svc_rbf_grid_model.predict(val_x)
print(confusion_matrix(val_y, svc_rbf_grid_predictions_val))
print(classification_report(val_y, svc_rbf_grid_predictions_val))

# 测试集
print('SVC_rbf_调参后最终测试集')
grid_predictions_test = svc_rbf_grid_model.predict(test_x)
print(confusion_matrix(test_y, grid_predictions_test))
print(classification_report(test_y, grid_predictions_test))

# plot_confusion_matrix(grid_model, test_x, test_y, values_format='d', display_labels=['malignant', 'benign'])
# plt.show()








# 构建随机森林模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 拟合模型
rf_classifier.fit(train_x, train_y)

print('rf_测试集')
rf_predictions_test = rf_classifier.predict(test_x)
print(confusion_matrix(test_y, rf_predictions_test))
print(classification_report(test_y, rf_predictions_test))

# 计算特征重要性
importances = rf_classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(train_x.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), [feature_remain[i] for i in indices], rotation=90)
plt.xlim([-1, train_x.shape[1]])
plt.tight_layout()
plt.savefig('feature_importances.png')

# 找到最重要和最不重要的决策树
most_important_tree_index = np.argmax(importances)
most_important_tree = rf_classifier.estimators_[most_important_tree_index]

worst_tree_index = np.argmin(importances)
worst_tree = rf_classifier.estimators_[worst_tree_index]

# 可视化最重要的决策树
plt.figure(figsize=(20, 10))
plot_tree(most_important_tree, filled=True, feature_names=feature_remain)
plt.savefig('most_important_tree.png')

# 可视化最不重要的决策树
plt.figure(figsize=(20, 10))
plot_tree(worst_tree, filled=True, feature_names=feature_remain)
plt.savefig('worst_tree.png')