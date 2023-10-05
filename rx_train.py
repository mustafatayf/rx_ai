import numpy as np

from rx_utils import get_data, add_awgn, show_train, check_data
from rx_models import dense_nn_bpsk, dense_nn_qpsk, dense_nn_deep, lstm_bpsk, gru_bpsk, save_mdl

batch_size = 1024
NoD = 3 * 10 ** 5
MODULATION = 'bpsk'  # bpsk, qpsk

X_i, y_i = get_data('data_N1e6_bpsk/tau0.50_snr7_bpsk', NoD=NoD)  # tauD.FF_snrDD_bpsk, tauD.FF_nonoise_bpsk
# X, y = get_data(name='data_N1e5_qpsk/tau0.70_snr10_qpsk', NoD=NoD)  # tauD.FF_snrDD_bpsk, tauD.FF_nonoise_bpsk
# model = dense_nn_bpsk()
# model = dense_nn_deep()
# model = dense_nn_qpsk()
# model = lstm_bpsk(batch_size=batch_size)
model = gru_bpsk(batch_size=batch_size)
print(model.summary())
confs = {
    'loss': model.loss,
    'optimizer_class_name': model.optimizer.__class__.__name__,
    'optimizer_config': model.optimizer.get_config(),
    'metrics': model.compiled_metrics._metrics,  # noqa
}
for k, v in confs.items():
    print(k, v)


# Xs = add_awgn(y*2-1, snr=10)

# data control
# check_data(rx_data=X, ref_bit=y, modulation='bpsk')
# check_data(rx_data=X, ref_bit=y, modulation='qpsk')

if 'lstm' in model.name or 'gru' in model.name:
    isi = 7
    # padding for initial and ending values
    Xp = np.append(np.zeros(isi), X_i, axis=0)
    Xp = np.append(Xp, np.zeros(isi), axis=0)
    assert Xp.size == X_i.size + 2*isi, 'error'

    ls_x = np.empty(shape=(X_i.size, 2*isi+1))
    for i in range(X_i.size):
        ls_x[i, :] = Xp[i:i+2*isi+1]

    # X = np.reshape(ls_x, (ls_x.shape[0], ls_x.shape[1], 1))
    X = ls_x
else:
    X = X_i
y = y_i.astype(np.float16)


# https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM
# wandb.init(entity='..-..', project='dl-..-..')
# tf.keras.backend.clear_session()
history = model.fit(X, y, validation_split=0.1, epochs=20, batch_size=batch_size)  # callbacks=[WandbCallback()]
# save_mdl(model)
# plot train process
show_train(history)

# results = model.evaluate(x_test, y_test, batch_size=128)
# y_pred = model.predict(X[:10, :])
# y_true = y[:10, :]

# asd.predict([1.2, -3.4, 5.4])

