SELECT 
    -- get the columns we need based on NVIDIA previous experiments
    article_id::INT AS ARTICLE_ID, 
    customer_id::VARCHAR AS customer_id,
    price::FLOAT AS price,
    sales_channel_id::INT  as sales_channel_id,
    t_dat::DATETIME as t_dat
FROM 
     read_parquet('../transactions_train')
ORDER BY t_dat ASC

