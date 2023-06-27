get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("heart_cleveland_upload.csv")
print(data.head())

for column in data.columns:
    columns_null = data[column].isnull().sum()
    if columns_null > 0:
        print(column + ': ' + str(columns_null)) 
# нет пропусков в данных


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        elif col == 'timestamp':
            df[col] = pd.to_datetime(df[col])
        elif str(col_type)[:8] != 'datetime':
            df[col] = df[col].astype("category")
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(start_mem, end_mem)
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2), 
          "Мб (минус", round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df

print(data.info())
data = reduce_mem_usage(data)
print(data.info())

corr_matrix = np.corrcoef(data.T)
sns.set(rc={'font.size':13, 'figure.figsize': (10,10)})
sns.heatmap(corr_matrix, cbar=True, annot=True, square=True, fmt='.2f', 
            yticklabels=data.columns, xticklabels=data.columns).set_title("Корреляция", fontsize=15)

data.hist(figsize=(16, 24), layout=(16, 8))

columns = data.drop(labels=['condition'], axis=1).columns
columns = list(columns)
print(columns)

data_train, data_test = train_test_split(data, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)

def regression_model(df, columns):
    y = df['condition']
    x = pd.DataFrame(df, columns=columns)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial')
    model.fit(x, y)
    return model


def logistic_regression(columns):
    x = pd.DataFrame(data_test, columns=columns)
    model = regression_model(data_train, columns)
    data_test['target'] = model.predict(x)
    return cohen_kappa_score(data_test['target'], data_test['condition'], weights='quadratic')


print('Логистическая регрессия:', round(logistic_regression(columns), 3))
