# %% Model Hyperparameter
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'ETH-USDT_GRU'
# make target OHLC to order column so we can drop V when we create dataset for y_batches
targets = ['closing_price','highest_price','lowest_price','open_price'] #must b a list
# Hyper-parameters
sequence_length = 128
pred_seq_length = 8
prediction_frequency = '1min'
num_epochs = 500
batch_size = 100
shuffle_batches = False

# define GRU with the following parameters
# input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
x_feature_dim = 5 # (OHLCV)
num_hidden_layers = 1
hidden_layer_width = 64
output_dimensions = len(targets)
dropout_probability = 0.3
learning_rate = .1
weight_decay = 1e-6

# GRUModel(x_feature_dim, hidden_layer_width, num_hidden_layers, output_dimensions, dropout_probability).to(device)
print('defined all hyperparams')

#%%
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

#%% Model training specs
# weight initialization
# https://pytorch.org/docs/stable/nn.init.html
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/57116138#57116138
def initialize_weights(model_to_initialize):
    for p in model_to_initialize.parameters():
        torch.nn.init.normal_(p, std=(1/(hidden_layer_width)))
    print('random-n inited weights')

model = GRUModel(x_feature_dim, hidden_layer_width, num_hidden_layers, pred_seq_length, output_dimensions, dropout_probability).to(device)
initialize_weights(model)
print(model.named_parameters())

criterion = nn.MSELoss()

# Adam
# https://arxiv.org/abs/1412.6980
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimal ADAM parameters: Good default settings for the tested machine learning problems are stepsize "lr" = 0.001,
#  Exponential decay rates for the moment estimates "betas" β1 = 0.9, β2 = 0.999 and
#  epsilon decay rate "eps" = 10−8

# AdamW decouple weight decay regularization improving upon standard Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# https://arxiv.org/abs/1711.05101

#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.08)
# An LBFGS solver is a quasi-Newton method which uses the inverse of the Hessian to estimate the curvature of the
# parameter space. In sequential problems, the parameter space is characterised by an abundance of long,
# flat valleys, which means that the LBFGS algorithm often outperforms other methods such as Adam, particularly when
# there is not a huge amount of data.

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,  verbose=True)
# try with Cosine Annealing LR
# torch.optim.lr_scheduler.CosineAnnealingLR

opt = Optimization(model=model, model_name=model_name, loss_fn=criterion, optimizer=optimizer)

opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=num_epochs, n_features=x_feature_dim)
