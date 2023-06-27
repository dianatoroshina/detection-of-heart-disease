import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
import lightgbm as lgb
from sklearn import preprocessing


data = pd.read_csv("heart_cleveland_upload.csv")
print(data.head())

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


data = reduce_mem_usage(data)
print(data.info())

columns = data.drop(labels=['condition'], axis=1).columns
columns = list(columns)
print(columns)

scaler = preprocessing.StandardScaler()
data_transformed = pd.DataFrame(scaler.fit_transform(pd.DataFrame(data, columns=columns)))
columns_transformed = data_transformed.columns
data_transformed['condition'] = data['condition']

data_train, data_test = train_test_split(data_transformed, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)

x = pd.DataFrame(data_train, columns=columns_transformed)
model = lgb.LGBMClassifier(random_state=17, max_depth=18, min_child_samples=19, num_leaves=34)

lgb_params = {
    'max_depth': range(16, 19),
    'num_leaves': range(34, 37),
    'min_child_samples': (17, 20)
}
grid = GridSearchCV(model, lgb_params, cv=5, n_jobs=4, verbose=True)
grid.fit(x, data_train['condition'])

print(grid.best_params_)
model = lgb.LGBMRegressor(random_state=17, max_depth=grid.best_params_['max_depth'],
                         min_child_samples=grid.best_params_['min_child_samples'],
                         num_leaves=grid.best_params_['num_leaves'], n_estimators=1000, objective='multiclass',
                         num_class=2)

model.fit(x, data_train['condition'])

def calculate_model(x):
    return np.argmax(model.predict([x]))


x_test = pd.DataFrame(data_test, columns=columns_transformed)
data_test['target'] = x_test.apply(calculate_model, axis=1, result_type='expand')

print('LightGBM:', round(cohen_kappa_score(data_test['target'], data_test['condition'], weights='quadratic'), 3))
