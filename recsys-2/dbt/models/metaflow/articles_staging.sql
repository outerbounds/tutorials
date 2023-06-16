SELECT 
    article_id::INT AS ARTICLE_ID, 
    product_code::INT AS PRODUCT_CODE, 
    product_type_no::INT AS PRODUCT_TYPE_NO, 
    product_group_name::VARCHAR AS PRODUCT_GROUP_NAME, 
    graphical_appearance_no::INT AS graphical_appearance_no, 
    colour_group_code::INT AS colour_group_code, 
    perceived_colour_value_id::INT AS perceived_colour_value_id, 
    perceived_colour_master_id::INT AS perceived_colour_master_id, 
    department_no::INT AS department_no,
    index_code::VARCHAR AS index_code, 
    index_group_no::INT AS index_group_no, 
    section_no::INT AS section_no, 
    garment_group_no::INT AS garment_group_no
FROM 
    read_parquet('../articles')