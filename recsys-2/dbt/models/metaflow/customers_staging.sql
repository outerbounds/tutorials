SELECT 
    COALESCE(TRY_CAST(ACTIVE AS DOUBLE), 0.0) AS ACTIVE,
    COALESCE(TRY_CAST(FN AS DOUBLE), 0.0) AS FN,
    COALESCE(TRY_CAST(age AS DOUBLE), 0.0) AS age,
    club_member_status,
    customer_id,
    fashion_news_frequency,
    postal_code
FROM (
    SELECT 
        -- get the columns we need based on NVIDIA previous experiments
        Active AS ACTIVE,
        FN,
        age,
        club_member_status::VARCHAR AS club_member_status,
        customer_id::VARCHAR AS customer_id,
        fashion_news_frequency::VARCHAR AS fashion_news_frequency,
        postal_code::VARCHAR AS postal_code
    FROM 
        read_parquet('../customers')
    ) a

