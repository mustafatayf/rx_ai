import wandb
import numpy as np
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from rx_utils import get_data, show_train, check_data
from rx_models import dense_nn_bpsk, dense_nn_qpsk, dense_nn_deep, lstm_bpsk, gru_bpsk, gru_qpsk, save_mdl

TAU = 0.80  # 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
SNR = 10  # 0, 1, 2, ..., 10, nonoise  # noqa
IQ = 'bpsk'  # bpsk, qpsk   # noqa

MODEL = 'gru'  # 'dense', 'lstm', 'gru'
model = ''

# train parameters
epochs = 40
batch_size = 1024
NoD = 10 ** 6
val_split = 0.1

# Load the training data
X_i, y_i = get_data(name='data_{iq}/tau{tau:.2f}_snr{snr}_{iq}'.format(iq=IQ, tau=TAU, snr=SNR), NoD=NoD)

# call the model if the model is not initialized yet
if model == '':
    model = eval(MODEL + '_' + IQ)(batch_size=batch_size)  # TODO fix security risk // CAUTION
    # model = gru_qpsk(batch_size=batch_size)

print(model.summary())
confs = {
    'loss': model.loss,
    'optimizer_class_name': model.optimizer.__class__.__name__,
    'optimizer_config': model.optimizer.get_config(),
    'metrics': model.compiled_metrics._metrics,  # noqa
}
for k, v in confs.items():
    print(k, v)

# DATA pre-processing

# Xs = add_awgn(y*2-1, snr=10)

# data control
# check_data(rx_data=X_i, ref_bit=y_i, modulation=IQ)

# TODO make time series generation as function
# single to time series data
if 'lstm' in model.name or 'gru' in model.name:
    isi = 7
    # padding for initial and ending values
    d = len(X_i.shape)
    assert d < 3, 'high dimensional input does not supported, only 1D or 2D'
    if d == 1:
        tmp_pad = np.zeros(isi)  # abs(X_i[:isi, :]*0)
    else:
        tmp_pad = abs(X_i[:isi, :]*0)
    Xp = np.concatenate((tmp_pad, X_i, tmp_pad), axis=0)

    sl = list(X_i.shape)
    sl.insert(1, 2 * isi + 1)
    ls_x = np.empty(shape=tuple(sl))
    if d == 1:
        for i in range(sl[0]):
            ls_x[i, :] = Xp[i:i + 2 * isi + 1]
    else:
        for i in range(sl[0]):
            ls_x[i, :, :] = Xp[i:i + 2 * isi + 1, :]

    # X = np.reshape(ls_x, (ls_x.shape[0], ls_x.shape[1], 1))
    X = ls_x
else:
    X = X_i

# update label type to float for evaluating performance metrics
y = y_i.astype(np.float16)

# Weight and Biases integration
# https://docs.wandb.ai/tutorials/keras_models
configs = dict(
    tau=TAU, snr=SNR,
    modulation=IQ,
    model=MODEL,
    batch_size=batch_size,
    data_size=len(y),
    validation_split=val_split,
    # learning_rate = 1e-3,
    epochs=epochs
)
# wandb.init(project='rx_ai',
#            config=configs
#            )
# tf.keras.backend.clear_session()
history = model.fit(X, y,
                    validation_split=val_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    # callbacks=[WandbMetricsLogger(log_freq=10), WandbModelCheckpoint('models/')]  # WandbCallback()
                    )

save_mdl(model, history=history)
# plot train process
show_train(history)

# wandb.finish()

# results = model.evaluate(x_test, y_test, batch_size=128)
# y_pred = model.predict(X[:10, :])
# y_true = y[:10, :]


# references

# https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM
