import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from torcheval.metrics import R2Score
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# %%
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
        
        ]

target_col = ['exret']

df = df[perm_cols + add_cols + target_col]
scaler = StandardScaler()
df[add_cols] = scaler.fit_transform(df[add_cols])

# %%
seq_len = 36
pred_len = 1
window_size = seq_len + pred_len

# %%
def generate_date_range(lower, upper, step):

    # List to hold the dates
    date_list = []

    # Generate dates with a 3-month step
    current_date = lower
    while current_date <= upper:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += relativedelta(months=step)
    
    return date_list

# pairs of date start and end bounds for masked time periods
first_start = datetime(2000, 4, 15)
last_start = datetime(2023, 12, 15)
start_list = generate_date_range(first_start, last_start, pred_len)
end_list = generate_date_range(first_start, last_start, pred_len)
pred_start_list = generate_date_range(datetime(2000, 4, 15) - relativedelta(months=seq_len), datetime(2020, 12, 15), pred_len)

# print(start_list)

# %%
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

# %%
test_data_df = df.copy()
train_data_df = df.copy().dropna()
pred_data_df = df.copy().dropna()

# for i in range(1):
for i in range(len(start_list)-1, -1, -1):

    print("processing masked date from {} to {}:".format(start_list[i], end_list[i]))
# %%
    train_df1 = train_data_df[train_data_df.date < start_list[i]]
    train_df2 = train_data_df[train_data_df.date > end_list[i] ]
    test_df = test_data_df.loc[(test_data_df.date >= start_list[i]) & (test_data_df.date <= end_list[i]), perm_cols + ['exret']]
    predict_df = pred_data_df[(pred_data_df.date >= pred_start_list[i]) & (pred_data_df.date < start_list[i])]

    # print(sorted(train_df1.date.unique()))
    # print(sorted(train_df2.date.unique()))
    # print(sorted(test_df.date.unique()))
    # print(sorted(predict_df.date.unique()))
    
    # only including funds with full pred len in test df
    test_fund_df = test_df.groupby('PRODUCTREFERENCE').agg({'date': 'count'}).reset_index()
    test_df = test_df[test_df.PRODUCTREFERENCE.isin(list(test_fund_df[test_fund_df.date >= pred_len]['PRODUCTREFERENCE']))]

    # inference performed on only funds present in test and have enough history
    count_df = predict_df.groupby(['PRODUCTREFERENCE']).agg({'date': 'count'}).reset_index()
    predict_df.merge(count_df.loc[count_df.date >= seq_len], on='PRODUCTREFERENCE') 
    funds_to_eval = list(predict_df.merge(test_df, on='PRODUCTREFERENCE', how='inner')['PRODUCTREFERENCE'].unique())
    predict_df = predict_df[predict_df.PRODUCTREFERENCE.isin(funds_to_eval)]
    predict_df['series_id'] = predict_df['PRODUCTREFERENCE']

    # if only including the products with enough history
    predict_fund_history_count = predict_df.groupby(['PRODUCTREFERENCE']).agg({'date':'count'}).reset_index()
    predict_funds = list(predict_fund_history_count.loc[predict_fund_history_count.date >= seq_len, 'PRODUCTREFERENCE'])
    predict_df = predict_df[predict_df.PRODUCTREFERENCE.isin(predict_funds)]

    # %%
    # Step 3: Create sliding windows of size seq len and stack them
    window_size = seq_len + pred_len

    windows = []

    for j in train_df1.PRODUCTREFERENCE.unique():
        temp_df = train_df1[train_df1.PRODUCTREFERENCE == j]
        # if len(temp_df) < 40:
        #     continue
        if len(temp_df) < window_size:
            continue

        sliding_windows = create_sliding_windows(temp_df, window_size)
        sliding_windows = sliding_windows.reset_index()
        sliding_windows['series_id'] = sliding_windows['PRODUCTREFERENCE'].astype("string") + '_0' + '_' +  \
                                    sliding_windows['level_0'].astype("string")
        sliding_windows = sliding_windows.drop(columns=['level_0', 'level_1'])
        windows.append(sliding_windows)

    for k in train_df2.PRODUCTREFERENCE.unique():
        temp_df = train_df2[train_df2.PRODUCTREFERENCE == k]
        # if len(temp_df) < 40:
        #     continue
        if len(temp_df) < window_size:
            continue

        sliding_windows = create_sliding_windows(temp_df, window_size)
        sliding_windows = sliding_windows.reset_index()
        sliding_windows['series_id'] = sliding_windows['PRODUCTREFERENCE'].astype("string") + '_1' + '_' +  \
                                    sliding_windows['level_0'].astype("string")
        sliding_windows = sliding_windows.drop(columns=['level_0', 'level_1'])
        windows.append(sliding_windows)

    # %%
    train_input_df = pd.concat(windows).reset_index(drop=True)
    train_input_df ['date'] = pd.to_datetime(train_input_df ['date'], format= '%Y-%m-%d' )

    # %%
    # static_features_df = train_input_df[['series_id', 'PRODUCTREFERENCE']].drop_duplicates().reset_index(drop=True)
    # predict_static_features = predict_df[['series_id', 'PRODUCTREFERENCE']].drop_duplicates().reset_index(drop=True)


    # %%
    train_data = TimeSeriesDataFrame.from_data_frame(
        train_input_df.drop(columns=['PRODUCTREFERENCE']),
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
        prediction_length=pred_len,
        path="benchmark/long_short_predictor_{}_{}".format(start_list[i], end_list[i]),
        target="exret",
        eval_metric="MASE",
        freq="MS",
        # verbosity=4,
    )

    # %%
    hyperparameters = {
        'TemporalFusionTransformer': {
            # 'context_length': 36  # Fixed look-back window of 5 periods
        },
        # 'RecursiveTabular': {},
        # 'DirectTabular':{},
        # 'Theta':{},
        # 'ETS':{},
    }

    # %%
    predictor.fit(
        train_data=train_data,
        # hyperparameters=hyperparameters,
        presets="medium_quality", 
    )

    # %%
    predictions = predictor.predict(predict_data)
    predictions.head()

    # %%
    pred_df = predictions.reset_index()
    pred_df['yearmonth'] = pred_df.timestamp.dt.strftime('%Y-%m')
    test_df['yearmonth'] = test_df.date.str.slice(0, -3)
    eval_df = pred_df.merge(test_df, how="inner", left_on=['item_id', 'yearmonth'], right_on=['PRODUCTREFERENCE', 'yearmonth'])

    print(eval_df.head(5))

    eval_df.to_csv('./long_short_pred_results/results_{}_{}.csv'.format(start_list[i], end_list[i]), index=False)
    # %%
    preds = list(eval_df['mean'])
    trues = list(eval_df['exret'])

    metrics = R2Score()
    # input = torch.tensor(preds[:,:,0].flatten())
    # target = torch.tensor(trues[:,:,0].flatten())

    input = torch.tensor(preds)
    target = torch.tensor(trues)

    metrics.update(input, target)
    print(metrics.compute())


