from rx_utils import get_data, add_awgn, show_train, check_data
from rx_models import dense_nn_bpsk, dense_nn_qpsk, dense_nn_deep, save_mdl

NoD = 3 * 10 ** 5
MODULATION = 'qpsk'  # bpsk, qpsk

# X, y = get_data('data_N1e6_bpsk/tau0.50_snr10_bpsk', NoD=NoD)  # tauD.FF_snrDD_bpsk, tauD.FF_nonoise_bpsk
X, y = get_data(name='data_N1e5_qpsk/tau0.70_snr10_qpsk', NoD=NoD)  # tauD.FF_snrDD_bpsk, tauD.FF_nonoise_bpsk
# model = dense_nn_bpsk()
# model = dense_nn_deep()
model = dense_nn_qpsk()
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
# Xs = X
# check for data
# print('#data: {NoD}\t#diff: {diff}'.format(NoD=NoD, diff=sum(abs((Xs > 0)*1 - y))))

# data control
# check_data(rx_data=X, ref_bit=y, modulation='qpsk')

history = model.fit(X, y, validation_split=0.2, epochs=20)
# save_mdl(model)
# plot train process
show_train(history)

# results = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(X[:10, :])
y_true = y[:10, :]

# asd.predict([1.2, -3.4, 5.4])

