"""

Unfortunately, the dataset is not properly formatted. We simply pass it through pandas
to get a clean CSV we can import in the Flow using duckdb.

"""

import pandas as pd
import random


def clean_dataset():
    df_playlist = pd.read_csv(
        'spotify_dataset.csv', 
        on_bad_lines='skip', 
        # if you want to get a smaller dataset, you can subsample at the source here
        # skiprows=lambda i: i>0 and random.random() > 0.50
        )
    # clean up the col names
    df_playlist.columns = df_playlist.columns.str.replace('"', '')
    df_playlist.columns = df_playlist.columns.str.replace('name', '')
    df_playlist.columns = df_playlist.columns.str.replace(' ', '')
    # add a row id
    df_playlist.insert(0, 'row_id', range(0, len(df_playlist)))
    # show the df
    print(df_playlist.head())
    # print the final lenght for the df befo
    print("Total rows: {}".format(len(df_playlist)))
    # dump to parquet (better than csv for duckdb)
    df_playlist.to_parquet('cleaned_spotify_dataset.parquet')
    print("All done\n\nSee you, space cowboy\n")

    return


if __name__ == "__main__":
    clean_dataset()