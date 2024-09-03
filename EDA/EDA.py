import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import src.dataset_transform as dt
from io import StringIO
import sys

def process_product_data(json_path):
    """
    Process product data from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing product data.

    Returns:
        pd.DataFrame: A DataFrame with unpacked category columns and calculated category levels.
    """
    # Reading Dataset
    products = pd.read_json(json_path)

    # Displaying the column info (limited to columns, non-null counts, and data types)
    buffer = StringIO()
    products.info(buf=buffer)
    s = buffer.getvalue()

    # Parse the DataFrame information
    lines = s.split('\n')
    info_data = []
    for line in lines:
        if line.startswith(' # ') or line.strip().startswith('<class') or line.strip().startswith('memory usage:'):
            continue
        elif 'dtypes:' in line:
            break
        else:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure there are enough parts to unpack
                non_null_count = parts[-3]
                dtype = parts[-1]
                column_name = " ".join(parts[:-3])  # Handle multi-word column names
                if non_null_count.replace(',', '').isdigit():  # Ensure non_null_count is numeric
                    info_data.append([column_name, non_null_count, dtype])

    # Create a DataFrame from the parsed information
    columns = ['Column', 'Non-Null Count', 'Dtype']
    info_df = pd.DataFrame(info_data, columns=columns)

    # Format and clean the DataFrame
    info_df['Non-Null Count'] = info_df['Non-Null Count'].apply(lambda x: x.replace(',', '')).astype(int)

    # Display the DataFrame information in a tabular format
    print("\n" + "="*50)
    print("Dataset Information:".center(50, "="))
    print("="*50)
    display(info_df)

    # Unpack the 'category' column (assuming a function `split_category_column` exists in dt module)
    result_df = dt.split_category_column(products)

    # Count the categories per each product
    result_df['category_level'] = result_df[['subcat1', 'subcat2', 'subcat3', 'subcat4', 'subcat5', 'subcat6', 'subcat7']].count(axis=1)

    # Describe the distribution of category levels
    count_categories = result_df['category_level']

    # Get the descriptive statistics
    desc_stats = products.describe()

    # Convert the descriptive statistics to a DataFrame for better formatting
    desc_df = pd.DataFrame(desc_stats).transpose()

    # Display the descriptive statistics in a tabular format
    print("\n" + "="*50)
    print("Category Level Descriptive Statistics:".center(50, "="))
    print("="*50)
    display(desc_df)
    print("\n")

    print("\n" + "="*100)
    print("First row of the result after counting category levels:".center(100, "="))
    print("="*100)
    display(result_df.head(1))  # Use display() from IPython.display for better formatting
    print("\n")

    return result_df


def plot_product_data(result_df):
    """
    Generate and display plots based on the processed product data.

    Args:
        result_df (pd.DataFrame): A DataFrame with unpacked category columns and calculated category levels.
    """

    print("\n" + "="*50)
    print("Plots:".center(50, "="))
    print( "="*50)


    # Count the frequency of each category level
    cat_counts = result_df['category_level'].value_counts()

    # Create the figure and axis objects for the category levels
    plt.figure(figsize=(8, 4), facecolor='lightgray')  # Change figure background color

    # Create the color gradient using the 'plasma' colormap
    colors = plt.cm.plasma(np.linspace(0, 1, len(cat_counts)))

    # Create the bar chart with the color gradient
    bars = plt.bar(cat_counts.index, cat_counts.values, edgecolor='black', color=colors)

    # Add labels and title
    plt.xlabel('Category Levels')
    plt.ylabel('Products')
    plt.title('Product Categories')

    # Add numbers on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}', ha='center', va='bottom')

    # Optimize the layout
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap

    # Display the plot
    plt.show()

    # Set up the figure and axes for the subcategories
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), facecolor='lightgray')  # Change figure background color
    fig.patch.set_facecolor('lightgray')  # Change figure background color if necessary

    # Plot histogram for each subcategory level
    subcats = ['subcat1_name', 'subcat2_name', 'subcat3_name', 'subcat4_name', 'subcat5_name', 'subcat6_name', 'subcat7_name']
    titles = ['Subcategory 1', 'Subcategory 2', 'Subcategory 3', 'Subcategory 4', 'Subcategory 5', 'Subcategory 6', 'Subcategory 7']

    for i, (subcat, title) in enumerate(zip(subcats, titles)):
        cate_counts = result_df[subcat].value_counts()
        total_prod = cate_counts.sum()
        cat_count = result_df[subcat].nunique()

        # Find the category with the highest count
        max_category = cate_counts.idxmax()
        max_count = cate_counts.max()

        ax = axs[i // 3, i % 3]
        ax.set_facecolor('lightblue')  # Change axes background color
        colors = plt.get_cmap('plasma')(np.linspace(0, 1, len(cate_counts)))

        bars = ax.bar(cate_counts.index, cate_counts.values, color=colors, edgecolor='black')
        ax.set_xlabel(title)
        ax.set_ylabel('Products')
        ax.set_title(f'{max_category} - {max_count} products \n (Total: {total_prod}, Categories: {cat_count})')
        ax.set_xticks([])  # Changed to set_xticks([]) to avoid unnecessary labels

        # Add numbers on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}', ha='center', va='bottom', fontsize=10, color='black')

    plt.show()


def process_product_data_transformed(json_path):
    """
    Process product data from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing product data.

    Returns:
        pd.DataFrame: A DataFrame with unpacked category columns and calculated category levels.
    """
    # Attempt to read JSON Dataset
    try:
        # If the JSON file contains multiple JSON objects separated by newlines, use lines=True
        products_T = pd.read_json(json_path, lines=True)
    except ValueError as e:
        # Handle the ValueError exception
        print(f"ValueError: {e}. Attempting to read the file manually.")
        # Manually read and load JSON to inspect it
        with open(json_path, 'r') as file:
            try:
                data = json.load(file)
                products_T = pd.json_normalize(data)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                return None  # Return None or handle the error as needed

    # Displaying the column info (limited to columns, non-null counts, and data types)
    buffer = StringIO()
    products_T.info(buf=buffer)
    s = buffer.getvalue()

    # Parse the DataFrame information
    lines = s.split('\n')
    info_data = []
    for line in lines:
        if line.startswith(' # ') or line.strip().startswith('<class') or line.strip().startswith('memory usage:'):
            continue
        elif 'dtypes:' in line:
            break
        else:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure there are enough parts to unpack
                non_null_count = parts[-3]
                dtype = parts[-1]
                column_name = " ".join(parts[:-3])  # Handle multi-word column names
                if non_null_count.replace(',', '').isdigit():  # Ensure non_null_count is numeric
                    info_data.append([column_name, non_null_count, dtype])

    # Create a DataFrame from the parsed information
    columns = ['Column', 'Non-Null Count', 'Dtype']
    info_df = pd.DataFrame(info_data, columns=columns)

    # Format and clean the DataFrame
    info_df['Non-Null Count'] = info_df['Non-Null Count'].apply(lambda x: x.replace(',', '')).astype(int)

    # Display the DataFrame information in a tabular format
    print("\n" + "="*50)
    print("Dataset Information:".center(50, "="))
    print("="*50)
    display(info_df)

    # List of expected subcategory columns
    subcategory_columns = ['subcat1_name', 'subcat2_name', 'subcat3_name', 'subcat4_name', 'subcat5_name']

    # Check if all subcategory columns are present
    missing_columns = [col for col in subcategory_columns if col not in products_T.columns]
    if missing_columns:
        raise KeyError(f"The following columns are missing from the dataset: {missing_columns}")

    # Get the descriptive statistics
    desc_stat = products_T.describe(include='all')  # include='all' to get stats for all columns

    # Convert the descriptive statistics to a DataFrame for better formatting
    desc_dfs = pd.DataFrame(desc_stat).transpose()

    # Display the descriptive statistics in a tabular format
    print("\n" + "="*50)
    print("Descriptive Statistics:".center(50, "="))
    print("="*50)
    display(desc_dfs)
    print("\n")

    # Count the categories per each product
    products_T['category_level'] = products_T[subcategory_columns].count(axis=1)

    print("\n" + "="*100)
    print("First row of the result after counting category levels:".center(100, "="))
    print("="*100)
    display(products_T.head(1))  # Use display() from IPython.display for better formatting
    print("\n")

    return products_T

def plot_product_data_tranformed(products_T):
    """
    Generate and display plots based on the processed product data.

    Args:
        result_df (pd.DataFrame): A DataFrame with unpacked category columns and calculated category levels.
    """

    print("\n" + "="*50)
    print("Plots:".center(50, "="))
    print( "="*50)


    # Count the frequency of each category level
    cat_counts = products_T['category_level'].value_counts()

    # Create the figure and axis objects for the category levels
    plt.figure(figsize=(8, 4), facecolor='lightgray')  # Change figure background color

    # Create the color gradient using the 'plasma' colormap
    colors = plt.cm.plasma(np.linspace(0, 1, len(cat_counts)))

    # Create the bar chart with the color gradient
    bars = plt.bar(cat_counts.index, cat_counts.values, edgecolor='black', color=colors)

    # Add labels and title
    plt.xlabel('Category Levels')
    plt.ylabel('Products')
    plt.title('Product Categories')

    # Add numbers on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}', ha='center', va='bottom')

    # Optimize the layout
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap

    # Display the plot
    plt.show()

    # Set up the figure and axes for the subcategories
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), facecolor='lightgray')  # Change figure background color
    fig.patch.set_facecolor('lightgray')  # Change figure background color if necessary

    # Plot histogram for each subcategory level
    subcats = ['subcat1_name', 'subcat2_name', 'subcat3_name', 'subcat4_name', 'subcat5_name']
    titles = ['Subcategory 1', 'Subcategory 2', 'Subcategory 3', 'Subcategory 4', 'Subcategory 5']

    for i, (subcat, title) in enumerate(zip(subcats, titles)):
        cate_counts = products_T[subcat].value_counts()
        total_prod = cate_counts.sum()
        cat_count = products_T[subcat].nunique()

        # Find the category with the highest count
        max_category = cate_counts.idxmax()
        max_count = cate_counts.max()

        ax = axs[i // 3, i % 3]
        ax.set_facecolor('lightblue')  # Change axes background color
        colors = plt.get_cmap('plasma')(np.linspace(0, 1, len(cate_counts)))

        bars = ax.bar(cate_counts.index, cate_counts.values, color=colors, edgecolor='black')
        ax.set_xlabel(title)
        ax.set_ylabel('Products')
        ax.set_title(f'{max_category} - {max_count} products \n (Total: {total_prod}, Categories: {cat_count})')
        ax.set_xticks([])  # Changed to set_xticks([]) to avoid unnecessary labels

        # Add numbers on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}', ha='center', va='bottom', fontsize=10, color='black')

    plt.show()

