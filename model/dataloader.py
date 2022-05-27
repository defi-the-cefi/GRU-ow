# Dataset specification

train_data_root_path = r'./'
train_file_name = r'ETH-USDT'+prediction_frequency+'.csv'
first_diff_levels_data = True

# https://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
training_data = Dataset_Custom(
    root_path=train_data_root_path,
    data_path=train_file_name,
    flag='train',
    scale = True,
    size=[sequence_length, x_feature_dim, pred_seq_length],
    features='M',
    timeenc=0,
    targets=targets,
    freq='t', # 'h': hourly, 't':minutely
    first_diff=first_diff_levels_data #
)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle_batches, drop_last=True)

val = Dataset_Custom(
    root_path=train_data_root_path,
    data_path=train_file_name,
    flag='val',
    scale = True,
    size=[sequence_length, x_feature_dim, pred_seq_length],
    features='M',
    timeenc=0,
    targets=targets,
    freq='t', # 'h': hourly, 't':minutely
    first_diff=first_diff_levels_data #
)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)

test = Dataset_Custom(
    root_path=train_data_root_path,
    data_path=train_file_name,
    flag='test',
    scale = True,
    size=[sequence_length, x_feature_dim, pred_seq_length],
    features='M',
    timeenc=0,
    targets=targets,
    freq='t', # 'h': hourly, 't':minutely
    first_diff=first_diff_levels_data #
)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# use this one for testing purposes size 1 batch
test_loader1 = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
print('generated required datasets')
