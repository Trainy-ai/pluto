import time

import pandas as pd

from .args import get_prefs, init_test, timer

TAG = 'table'


@timer
def test_table(mlop, run, NUM_EPOCHS=None, ITEM_PER_EPOCH=None):
    if NUM_EPOCHS is None or ITEM_PER_EPOCH is None:
        NUM_EPOCHS = get_prefs(TAG)['NUM_EPOCHS']
        ITEM_PER_EPOCH = get_prefs(TAG)['ITEM_PER_EPOCH']

    WAIT_INT = 10
    WAIT = ITEM_PER_EPOCH * 0.0005

    for i in range(NUM_EPOCHS):
        table_data_str = mlop.Table(
            columns=['a', 'b'], data=[['a1', 'b1'], ['a2', 'b2']]
        )
        table_data_int = mlop.Table(
            columns=[i for i in range(10)],
            rows=[i for i in range(10)],
            data=[[i * j for i in range(10)] for j in range(10)],
        )
        table_data_float = mlop.Table(
            rows=['1.0', '2.0'], data=[[1.0, 2.0], [3.0, 4.0]]
        )
        table_pd = mlop.Table(
            dataframe=pd.DataFrame({'0': ['df1', 'df2'], '1': ['df3', 'df4']})
        )
        run.log(
            {
                f'{TAG}/str-{i}': table_data_str,
                f'{TAG}/int-{i}': table_data_int,
                f'{TAG}/float-{i}': table_data_float,
                f'{TAG}/pd-{i}': table_pd,
            }
        )
        if i % WAIT_INT == 0:
            print(f'{TAG}: Epoch {i + 1} / {NUM_EPOCHS}, sleeping {WAIT} seconds')
            time.sleep(WAIT)


if __name__ == '__main__':
    mlop, run = init_test(TAG)
    test_table(mlop, run)
