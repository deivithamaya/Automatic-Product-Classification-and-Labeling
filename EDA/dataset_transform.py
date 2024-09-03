import pandas as pd
import numpy as np




def split_category_column(products):
    """
    Splits the 'category' column in the products DataFrame into subcategories and 
    extracts 'id' and 'name' for each subcategory.

    Args:
        products (pd.DataFrame): DataFrame containing product information including a 'category' column.

    Returns:
        pd.DataFrame: A new DataFrame with subcategory details and original product information, 
                      excluding specified columns.
    """

    #products = pd.read_json(json_path)

    # Convert the 'category' column from list of dictionaries to a DataFrame
    products_cat = products['category'].apply(pd.Series)
    
    # Rename columns to have more meaningful names
    products_cat.columns = [f'subcat{i+1}' for i in range(products_cat.shape[1])]
    
    # Combine with the original DataFrame
    products_category = pd.concat([products, products_cat], axis=1)

    # Create a copy with only subcategory columns
    products_subcat = products_category[['subcat1', 'subcat2', 'subcat3', 'subcat4', 'subcat5', 'subcat6', 'subcat7']].copy()
    products_subcat.columns = [f'subcat{i+1}' for i in range(products_subcat.shape[1])]

    # Extract 'id' and 'name' values into new columns
    for col in products_subcat.columns:
        products_subcat[f'{col}_id'] = products_subcat[col].apply(lambda x: x['id'] if pd.notnull(x) else None)
        products_subcat[f'{col}_name'] = products_subcat[col].apply(lambda x: x['name'] if pd.notnull(x) else None)

    # Remove original subcategory columns from the combined DataFrame to avoid duplication
    products_category_clean = products_category.drop(columns=['subcat1', 'subcat2', 'subcat3', 'subcat4', 'subcat5', 'subcat6', 'subcat7'])

    # Combine the cleaned original DataFrame with the subcategory DataFrame
    products_subcategory = pd.concat([products_category_clean, products_subcat], axis=1)

    
        
    return products_subcategory

def rename_images(products_subcategory):
    """
    Renombra las imágenes en la columna 'image' utilizando el valor de la columna 'sku' 
    y agrega la extensión '.jpg'.
    
    Parameters:
    df (pandas.DataFrame): DataFrame con las columnas 'sku' e 'image'.
    
    Returns:
    pandas.DataFrame: DataFrame con la columna 'image' renombrada.
    """
    products_subcategory['image'] = products_subcategory['sku'].astype(str) + '.jpg'
    return products_subcategory

def drop_col_notNesesary(products_subcategory):
     # Drop specified columns
    result_droped = products_subcategory.drop(columns=[
        'sku', 'price', 'upc', 'category', 'shipping', 'manufacturer', 'model', 'url',
        'subcat1', 'subcat2', 'subcat3', 'subcat4', 'subcat5', 'subcat6', 'subcat7',
        'subcat1_id', 'subcat2_id', 'subcat3_id', 'subcat4_id', 'subcat5_id', 'subcat6_id',
        'subcat7_id', 'subcat6_name',
        'subcat7_name'
    ])

    return result_droped


def replace_to_other(result_droped):
    """
    Replaces rare categories in the columns 'subcat1_name', 'subcat2_name',
    'subcat3_name', 'subcat4_name', and 'subcat5_name' with 'Other', using a threshold
    of 1% of the total count in each column.
    
    Args:
    products_subcategory (pd.DataFrame): The DataFrame in which changes will be made.
    
    Returns:
    pd.DataFrame: The DataFrame with rare categories replaced by 'Other'.
    """
    dataset_F = result_droped.copy()
    
    # Define columns to process
    columns_to_process = ['subcat1_name', 'subcat2_name', 'subcat3_name', 'subcat4_name', 'subcat5_name']
    
    # Calculate thresholds
    thresholds = {
        col: dataset_F[col].count() * 0.01 for col in columns_to_process
    }
    
    for col in columns_to_process:
        threshold = thresholds[col]
        
        # Count the frequency of each category in the current column
        subcat_counts = dataset_F[col].value_counts()
        
        # Identify the categories that meet the condition
        rare_cats = subcat_counts[subcat_counts <= threshold].index
        
        # Replace values in the current column
        dataset_F[col] = dataset_F[col].apply(lambda x: 'Other' if x in rare_cats else x)
    
    return dataset_F




def transform_dataset(json_path):
    products = pd.read_json(json_path)
    
    products_subcategory= split_category_column(products)
    products_subcategory = rename_images(products_subcategory)

    result_droped = drop_col_notNesesary(products_subcategory) # type: ignore
    
    dataset_F= replace_to_other(result_droped)
    
    return dataset_F


