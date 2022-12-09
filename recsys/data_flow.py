from metaflow import FlowSpec, step, S3, Parameter, current

class DataFlow(FlowSpec):

    IS_DEV = Parameter(
        name='is_dev',
        help='Flag for dev development, with a smaller dataset',
        default='1'
    )
    
    @step
    def start(self):
        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self):
        """
        Get the data in the right shape by reading the parquet dataset
        and using DuckDB SQL-based wrangling to quickly prepare the datasets for
        training our Recommender System.
        """
        import duckdb
        import numpy as np
        # highlight-next-line
        con = duckdb.connect(database=':memory:')
        # highlight-start
        con.execute("""
            CREATE TABLE playlists AS 
            SELECT *, 
            CONCAT (user_id, '-', playlist) as playlist_id,
            CONCAT (artist, '|||', track) as track_id,
            FROM 'cleaned_spotify_dataset.parquet'
            ;
        """)
        # highlight-end
        # highlight-next-line
        con.execute("SELECT * FROM playlists LIMIT 1;")
        print(con.fetchone())
        tables = ['row_id', 'user_id', 'track_id', 'playlist_id', 'artist']
        for t in tables:
            # highlight-next-line
            con.execute("SELECT COUNT(DISTINCT({})) FROM playlists;".format(t))
            print("# of {}".format(t), con.fetchone()[0])
        sampling_cmd = ''
        if self.IS_DEV == '1':
            print("Subsampling data, since this is DEV")
            # highlight-next-line
            sampling_cmd = ' USING SAMPLE 10 PERCENT (bernoulli)'
        # highlight-start
        dataset_query = """
            SELECT * FROM
            (   
                SELECT 
                    playlist_id,
                    LIST(artist ORDER BY row_id ASC) as artist_sequence,
                    LIST(track_id ORDER BY row_id ASC) as track_sequence,
                    array_pop_back(LIST(track_id ORDER BY row_id ASC)) as track_test_x,
                    LIST(track_id ORDER BY row_id ASC)[-1] as track_test_y
                FROM 
                    playlists
                GROUP BY playlist_id 
                HAVING len(track_sequence) > 2
            ) 
            {}
            ;
            """.format(sampling_cmd)
        con.execute(dataset_query)
        df = con.fetch_df()
        # highlight-end
        print("# rows: {}".format(len(df)))
        print(df.iloc[0].tolist())
        con.close()
        train, validate, test = np.split(
            df.sample(frac=1, random_state=42), 
            [int(.7 * len(df)), int(.9 * len(df))])
        self.df_dataset = df
        self.df_train = train
        self.df_validate = validate
        self.df_test = test
        print("# testing rows: {}".format(len(self.df_test)))
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    DataFlow()
