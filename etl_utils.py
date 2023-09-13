import os
import pandas as pd
import numpy as np
import pprint
from sklearn.preprocessing import MinMaxScaler
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures
import itertools
from decimal import Decimal


def windows(data, window_size, step):
    r = np.arange(len(data))
    s = r[::step]
    z = list(zip(s, s + window_size))
    f = '{0[0]}:{0[1]}'.format
    g = lambda t: data.iloc[t[0]:t[1]]
    return pd.concat(map(g, z), keys=map(f, z))


def get_feature_embedding_for_window(df, stock, split=64):
    ts_name = f"{stock.strip('.csv')}_{str(df.Date.min())}_{str(df.Date.max())}"
    scaler=MinMaxScaler()
    df[['Open', 'Close']] = scaler.fit_transform(df[['Open', 'Close']])
    if (df['Open'].values == 0).all():
        df['Open'] = df['Close']

    prices = df[['Open', 'Close']].values.tolist()
    flat_values = [item for sublist in prices for item in sublist]
    df = df.rename(columns={"Date":"time"}) 
    ts_df = pd.DataFrame({'time':df.time.repeat(2), 
                          'price':flat_values})
    ts_df.drop_duplicates(keep='first', inplace=True)  
    pre_flat_values = flat_values[:split*2]
    post_flat_values = flat_values[split*2:]

    # Use Kats to extract features for the time window
    try:
        if not (len(np.unique(ts_df.price.tolist())) == 1 \
            or len(np.unique(ts_df.price.tolist())) == 0):
            timeseries = TimeSeriesData(ts_df[:split*2])
            features = TsFeatures().transform(timeseries)
            feature_list = [float(v) if not pd.isnull(v) else float(0) for _, v in features.items()]
            feature_list = [v / (split*2.0) if v > 1.0 else v for v in feature_list]
            if Decimal('Infinity') in feature_list or Decimal('-Infinity') in feature_list:
                return None, None
            feature_list = pre_flat_values + feature_list
            return (ts_name, feature_list), (ts_name, post_flat_values)
    except np.linalg.LinAlgError as e:
        print(f"Can't process {ts_name}:{e}")
    return None, None


def get_simple_pair_for_window(df, stock, split=64):
    ts_name = f"{stock.strip('.csv')}_{str(df.Date.min())}_{str(df.Date.max())}"
    prices = df[['Open', 'Close']].values.tolist()
    flat_values = [item for sublist in prices[:split*2] for item in sublist]
    post_flat_values = [item for sublist in prices[split*2:] for item in sublist]
    return (ts_name, flat_values), (ts_name, post_flat_values)


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def create_sliding_window_feature(stock, create_pair_func, window_size=64, post_window_size=16, step=10):
    print(f'Stock: {stock.strip(".csv")}')
    data = pd.read_csv(os.path.join('./data/stocks', stock))
    data = data.sort_index(axis=0, ascending=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.date

    # Interpolate data for missing dates
    data.set_index('Date', inplace=True)
    data = data.reindex(pd.date_range(start=data.index.min(),
                                        end=data.index.max(),
                                        freq='1D'))
    data = data.interpolate(method='linear')
    data = data.reset_index().rename(columns={'index': 'Date'})
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    
    # Create sliding windows dataset
    total_window_size = window_size + post_window_size
    wdf = windows(data, total_window_size, step)
    
    # Prepare sequences for upload 
    feature_pairs = []
    feature_post_pairs = []
    for window, new_df in wdf.groupby(level=0):
        if new_df.shape[0] == total_window_size:
            pair, post = create_pair_func(new_df, stock, split=window_size)
            if pair:
                feature_pairs.append(pair)
                feature_post_pairs.append(post)


    print(f'{len(feature_pairs)} new feature pairs')
    return feature_pairs, feature_post_pairs


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--post_window_size', type=int, default=16)
    parser.add_argument('--step', type=int, default=32)
    parser.add_argument('--folder_name', type=str, default='./data/stocks')
    parser.add_argument('--num_stocks', type=int, default=1)

    args = parser.parse_args()
    print(args)

    stocks = sorted(os.listdir(args.folder_name))
    total_feature_pairs = []
    total_feature_post_pairs = []

    for stock in stocks[:args.num_stocks]:
        feature_pairs, feature_post = create_sliding_window_feature(stock,
                                                                    get_feature_embedding_for_window,
                                                                    # get_simple_pair_for_window,
                                                                    window_size=args.window_size,
                                                                    post_window_size=args.post_window_size,
                                                                    step=args.step)
        total_feature_pairs += feature_pairs
        total_feature_post_pairs += feature_post

    print(len(total_feature_pairs))


if __name__ == '__main__':
    main()

