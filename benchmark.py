import pandas as pd
import numpy as np
import torch
from torcheval.metrics import R2Score
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

seq_len = 36
pred_len = 3

df = pd.read_csv("dataset/hf/hf.csv")

perm_cols = ['date', 'PRODUCTREFERENCE']
add_cols = [ 
            # 'ret',
        'aum', 
        'SENT_', 
        # 'PTFSBD', 
        # 'PTFSFX', 
        'PTFSCOM', 
        # 'em', 
        'sp500', 
        'sizespread', 
        'bondmkt', 
        # 'creditspread',
        'SMB', 'HML','RF', 'mom', 
        # 'con','ipg','tfp','term','def','dei','mkt','lab',
        'confeature', 'tfpfeature', 'ipgfeature', 'termfeature', 'deffeature', 'deifeature', 'mktfeature', 'labfeature',
        # 'ret',
        'exret',
        ]

df = df[perm_cols + add_cols]

dataset = df.copy()
dataset = dataset.dropna()
train_df = dataset[dataset.date <= '2023-09-15']
test_df = dataset[dataset.date >= '2023-10-15']

test_fund_df = test_df.groupby('PRODUCTREFERENCE').agg({'date': 'count'}).reset_index()
test_df = test_df[test_df.PRODUCTREFERENCE.isin(list(test_fund_df[test_fund_df.date >= 3]['PRODUCTREFERENCE']))]

predict_df = dataset[(dataset.date >= '2020-10-15') & ( dataset.date <= '2023-09-15')]
count_df = predict_df.groupby(['PRODUCTREFERENCE']).agg({'date': 'count'}).reset_index()
predict_df.merge(count_df.loc[count_df.date >= seq_len], on='PRODUCTREFERENCE') 
funds_to_eval = list(predict_df.merge(test_df, on='PRODUCTREFERENCE', how='inner')['PRODUCTREFERENCE'].unique())
predict_df = predict_df[predict_df.PRODUCTREFERENCE.isin(funds_to_eval)]
predict_df['series_id'] = predict_df['PRODUCTREFERENCE']

# if only including the products with enough history
predict_fund_history_count = predict_df.groupby(['PRODUCTREFERENCE']).agg({'date':'count'}).reset_index()
predict_funds = list(predict_fund_history_count.loc[predict_fund_history_count.date >= seq_len, 'PRODUCTREFERENCE'])
predict_df = predict_df[predict_df.PRODUCTREFERENCE.isin(predict_funds)]

def create_sliding_windows(data, window_size):
    windows = []
    
    if len(data) < window_size:
        window = data.copy()
        window.reset_index(drop=True, inplace=True)
        windows.append(window)
    else:
        for i in range(len(data) - window_size + 1):
            window = data.iloc[i:i+window_size].copy()
            window.reset_index(drop=True, inplace=True)
            windows.append(window)
    return pd.concat(windows, keys=range(len(windows)))

# Step 3: Create sliding windows of size 12 and stack them
window_size = seq_len + pred_len

windows = []

for i in train_df.PRODUCTREFERENCE.unique():
    temp_df = train_df[train_df.PRODUCTREFERENCE == i]
    # if len(temp_df) < 40:
    #     continue
    if len(temp_df) < window_size:
        continue

    sliding_windows = create_sliding_windows(temp_df, window_size)
    sliding_windows = sliding_windows.reset_index()
    sliding_windows['series_id'] = sliding_windows['PRODUCTREFERENCE'].astype("string")  + '_' +  \
                                   sliding_windows['level_0'].astype("string")
    sliding_windows = sliding_windows.drop(columns=['level_0', 'level_1'])
    windows.append(sliding_windows)

train_data_df = pd.concat(windows).reset_index(drop=True)
train_data_df['date'] = pd.to_datetime(train_data_df['date'], format= '%Y-%m-%d' )

# static_features_df = train_data_df[['series_id', 'PRODUCTREFERENCE']].drop_duplicates().reset_index(drop=True)
# predict_static_features = predict_df[['series_id', 'PRODUCTREFERENCE']].drop_duplicates().reset_index(drop=True)

train_data = TimeSeriesDataFrame.from_data_frame(
    train_data_df.drop(columns=['PRODUCTREFERENCE']),
    id_column="series_id",
    timestamp_column="date", 
    # static_features_df=static_features_df,
)

predict_data = TimeSeriesDataFrame.from_data_frame(
    predict_df.drop(columns=['PRODUCTREFERENCE']),
    id_column="series_id",
    timestamp_column="date", 
    # static_features_df=predict_static_features,
)

predictor = TimeSeriesPredictor(
    prediction_length=3,
    path="benchmark/autogluon-48-3-monthly",
    target="exret",
    eval_metric="MASE",
    freq="MS",
    # verbosity=4,
)

hyperparameters = {
    'TemporalFusionTransformer': {
        # 'context_length': 36  # Fixed look-back window of 5 periods
    },
    # 'RecursiveTabular': {},
    # 'DirectTabular':{},
    # 'Theta':{},
    # 'ETS':{},


    
}


predictor.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    # presets="medium_quality",
    
)

predictions = predictor.predict(predict_data)

pred_df = predictions.reset_index()
pred_df['yearmonth'] = pred_df.timestamp.dt.strftime('%Y-%m')
test_df['yearmonth'] = test_df.date.str.slice(0, -3)
eval_df = pred_df.merge(test_df, how="inner", left_on=['item_id', 'yearmonth'], right_on=['PRODUCTREFERENCE', 'yearmonth'])

preds = list(eval_df['mean'])
trues = list(eval_df['exret'])

metrics = R2Score()
# input = torch.tensor(preds[:,:,0].flatten())
# target = torch.tensor(trues[:,:,0].flatten())

input = torch.tensor(preds)
target = torch.tensor(trues)

metrics.update(input, target)
print(metrics.compute())