#导入相关库
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#导入数据并提取text文本
train=pd.read_csv('train_set.csv',sep='\t')
test=pd.read_csv('test_a.csv',sep='\t')
train.shape
(200000,2)
train[:5]
train_t=train['text']
test_t=test['text']
#用TF-IDF对文本提取特征并向量化
ti=TfidfVectorizer(max_features=6000).fit(train_t.values)
train_tf=ti.transform(train_t.values)
train_tf.shape
(200000,6000)
test_tf=ti.transform(test_t.values)
x_train=train_tf
y_train=train['label']
x_text=test_tf
#使用LR模型进行训练和验证
clf=LogisticRegression(C=10,multi_class='ovr',solver='liblinear',class_weight='balanced')
clf.fit(x_train,y_train.values)
val_pred=clf.predict(x_train[10000:])
f1=f1_score(y_train.values[10000:],val_pred,average='macro')
print(f1)
#测试集
df=pd.DataFrame()
df['label']=clf.predict(x_text)
df.to_csv('submit1.csv',index=None)