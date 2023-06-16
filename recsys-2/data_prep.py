from metaflow import FlowSpec, step, batch, Parameter, current
from datetime import datetime

class DataPrepFlow(FlowSpec):
    
    ### DATA PARAMETERS ###
    ROW_SAMPLING = Parameter(
        name='row_sampling',
        help='Row sampling: if 0, NO sampling is applied. Needs to be an int between 1 and 100',
        default='1'
    )

    # NOTE: data parameters - we split by time, leaving the last two weeks for validation and tests
    # The first date in the table is 2018-09-20
    # The last date in the table is 2020-09-22
    TRAINING_END_DATE = Parameter(
        name='training_end_date',
        help='Data up until this date is used for training, format yyyy-mm-dd',
        default='2020-09-08'
    )

    VALIDATION_END_DATE = Parameter(
        name='validation_end_date',
        help='Data up after training end and until this date is used for validation, format yyyy-mm-dd',
        default='2020-09-15'
    )

    @step
    def start(self):
        """
        Start-up: check everything works or fail fast!
        """
        # print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        # we need to check if Metaflow is running with remote (s3) data store or not
        from metaflow.metaflow_config import DATASTORE_SYSROOT_S3 
        print("DATASTORE_SYSROOT_S3: %s" % DATASTORE_SYSROOT_S3)
        if DATASTORE_SYSROOT_S3 is None:
            print("ATTENTION: LOCAL DATASTORE ENABLED")
        # check variables and connections are working fine
        assert int(self.ROW_SAMPLING)
        # check the data range makes sense
        self.training_end_date = datetime.strptime(self.TRAINING_END_DATE, '%Y-%m-%d')
        self.validation_end_date = datetime.strptime(self.VALIDATION_END_DATE, '%Y-%m-%d')
        assert self.validation_end_date > self.training_end_date

        self.next(self.get_dataset)

    @step
    def get_dataset(self):
        """
        Get the data in the right shape using duckDb, after the dbt transformation
        """
        from pyarrow import Table as pt
        import duckdb
        # check if we need to sample - this is useful to iterate on the code with a real setup
        # without reading in too much data...
        _sampling = int(self.ROW_SAMPLING)
        sampling_expression = '' if _sampling == 0 else 'USING SAMPLE {} PERCENT (bernoulli)'.format(_sampling)
        # thanks to our dbt preparation, the ML models can read in directly the data without additional logic
        query = """
            SELECT 
                ARTICLE_ID,
                PRODUCT_CODE, 
                PRODUCT_TYPE_NO,
                PRODUCT_GROUP_NAME,
                GRAPHICAL_APPEARANCE_NO,
                COLOUR_GROUP_CODE,
                PERCEIVED_COLOUR_VALUE_ID,
                PERCEIVED_COLOUR_MASTER_ID,
                DEPARTMENT_NO,
                INDEX_CODE,
                INDEX_GROUP_NO,
                SECTION_NO,
                GARMENT_GROUP_NO,
                ACTIVE,
                FN,
                AGE,
                CLUB_MEMBER_STATUS,
                CUSTOMER_ID,
                FASHION_NEWS_FREQUENCY,
                POSTAL_CODE,
                PRICE,
                SALES_CHANNEL_ID,
                T_DAT
            FROM
                read_parquet('filtered_dataframe.parquet')
                {}
            ORDER BY
                T_DAT ASC
        """.format(sampling_expression)
        print("Fetching rows with query: \n {} \n\nIt may take a while...\n".format(query))
        # fetch raw dataset
        con = duckdb.connect(database=':memory:')
        con.execute(query)
        dataset = con.fetchall()
        # convert the COLS to lower case (Keras does complain downstream otherwise)
        cols = [c[0].lower() for c in con.description]
        dataset = [{ k: v for k, v in zip(cols, row) } for row in dataset]
        # debug
        print("Example row", dataset[0])
        self.item_id_2_meta = { str(r['article_id']): r for r in dataset }
        # we split by time window, using the dates specified as parameters
        # NOTE: we could actually return Arrow table directly, by then running three queries over
        # a different date range (e.g. https://duckdb.org/2021/12/03/duck-arrow.html)
        # For simplicity, we kept here the original flow compatible with warehouse processing
        train_dataset = pt.from_pylist([row for row in dataset if row['t_dat'] < self.training_end_date])
        validation_dataset = pt.from_pylist([row for row in dataset 
            if row['t_dat'] >= self.training_end_date and row['t_dat'] < self.validation_end_date])
        test_dataset = pt.from_pylist([row for row in dataset if row['t_dat'] >= self.validation_end_date])
        print("# {:,} events in the training set, {:,} for validation, {:,} for test".format(
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset)
        ))
        # store and version datasets as a map label -> datasets, for consist processing later on
        self.label_to_dataset = {
            'train': train_dataset,
            'valid': validation_dataset,
            'test': test_dataset
        }
        # go to the next step for NV tabular data
        self.next(self.end)
    
    @step
    def end(self):
        """
        Just say bye!
        """
        print("All done\n\nSee you, recSys cowboy\n")
        return

if __name__ == '__main__':
    DataPrepFlow()
