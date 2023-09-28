from rx_utils import get_data, add_awgn, show_train
from rx_models import dense_nn_bpsk, save_mdl

NoD = 10 ** 5

X, y = get_data('tau1_snr10_bpsk', NoD=NoD)
model = dense_nn_bpsk()
print(model.summary())
confs = {
    'loss': model.loss,
    'optimizer_class_name': model.optimizer.__class__.__name__,
    'optimizer_config': model.optimizer.get_config(),
    'metrics': model.compiled_metrics._metrics,  # noqa
}
for k, v in confs.items():
    print(k, v)


Xs = add_awgn(y*2-1, snr=10)
# Xs = X
# check for data
print('#data: {NoD}\t#diff: {diff}'.format(NoD=NoD, diff=sum(abs((Xs > 0)*1 - y))))

history = model.fit(Xs, y, validation_split=0.2, epochs=40)
save_mdl(model)
# plot train process
show_train(history)

# results = model.evaluate(x_test, y_test, batch_size=128)
# y_pred = model.predict(x_)

# asd.predict([1.2, -3.4, 5.4])

