"""

We do some dataset wrangling to get final parquet files for our flow.

Data comes from H and M dataset: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

Download and unzip the csv files in this folder before running the cleaning script:

* articles.csv
* customers.csv
* transactions_train.csv

After transformation, we run a sample query to verify data is readable. You can comment the function
out if you don't wish to run the test.

"""

import time
import pandas as pd
import random
import sys

def create_final_dataset(tables: list, sample_fraction: float = 0.25):
    for table in tables:
        skip_function = None if table != 'transactions_train' else lambda i: i>0 and random.random() > sample_fraction
        _df = pd.read_csv(
            '{}.csv'.format(table), 
            on_bad_lines='skip',
            skiprows=skip_function)
        # show the df
        print(_df.head())
        print(_df.dtypes)
        # print the lenght
        print("Total rows for {}: {}\n\n".format(table, len(_df)))
        # dump to parquet (better than csv for duckdb)
        _df.to_parquet('{}'.format(table))

    return
    

def run_duckdb_test():
    import duckdb
    # query counts for a few days, months, year
    start = time.time()
    con = duckdb.connect()
    # query different time spans
    end_dates = ['2018-09-22', '2018-10-20', '2019-10-22']
    for d in end_dates:
        print(duckdb.query('''
            SELECT 
                COUNT(customer_id)
            FROM 
                read_parquet('transactions_train')
            WHERE 
                t_dat BETWEEN '2018-09-20' AND '2018-9-22'
        ''').fetchall())
        print("Elapsed time (s) for {}: {}".format(d, time.time() - start))

    return


if __name__ == "__main__":
    TABLES = [
        'articles',
        'customers',
        'transactions_train'
    ]
    create_final_dataset(TABLES, float(sys.argv[1]))
    run_duckdb_test()
    print("All done\n\nSee you, space cowboy\n")