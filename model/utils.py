import numpy as np
import pandas as pd


def check_subcategory(category: pd.DataFrame, subcategory_name: str, parent_name: str):
    if category['name'] == parent_name:
        for subcategory in category.get('subCategories', []):
            if subcategory['name'] == subcategory_name:
                return True
    for subcategory in category.get('subCategories', []):
        if check_subcategory(subcategory, subcategory_name, parent_name):
            return True
    return False


def is_category_valid(category_name: str, categories: pd.DataFrame):
        """
        Check if a category is valid.

        Args:
            category_name (str): The name of the category to be checked.

        Returns:
            bool: True if the category is valid, False otherwise.
        """
        return category_name in categories['name'].values


def is_subcategory_valid(parent_name: str, subcategory_name: str, categories: pd.DataFrame):
    """
    Check if a subcategory is valid within a parent category.

    Args:
        parent_name (str): The name of the parent category.
        subcategory_name (str): The name of the subcategory to be checked.

    Returns:
        bool: True if the subcategory is valid within the parent category, False otherwise.
    """
    # Find the parent category in the DataFrame
    parent_category = categories[categories['name'] == parent_name]
    if not parent_category.empty:
        parent_category = parent_category.iloc[0]
        return check_subcategory(parent_category, subcategory_name, parent_name)
    return False

def _get_categories():
    cat_level_1 = [
            "Appliances","Audio","Cameras & Camcorders","Car Electronics & GPS","Cell Phones","Computers & Tablets",
            "Connected Home & Housewares","Health, Fitness & Beauty","Musical Instruments","Other","TV & Home Theater",
            "Toys, Games & Drones","Video Games"
        ]

    cat_level_2 = [
            "Appliance Parts & Accessories","Bluetooth & Wireless Speakers","Car Audio","Car Installation Parts & Accessories"
            "Cell Phone Accessories","Computer Accessories & Peripherals","Connected Home","Digital Camera Accessories","Furniture & Decor",
            "Headphones","Heating, Cooling & Air Quality","Home Audio","Housewares","Musical Instrument Accessories","Office & School Supplies","Other",
            "Personal Care & Beauty","Pre-Owned Games","Ranges, Cooktops & Ovens","Refrigerators","Sheet Music & DVDs","Small Kitchen Appliances",
            "TV & Home Theater Accessories","TV Stands, Mounts & Furniture","iPad & Tablet Accessories"
        ]

    cat_level_3 = [
            "A/V Cables & Connectors","Adapters, Cables & Chargers","All Laptops","All Refrigerators","Car Audio Installation Parts",
            "Cases, Covers & Keyboard Folios","Cell Phone Batteries & Power","Cell Phone Cases & Clips""Coffee, Tea & Espresso",
            "Cookware, Bakeware & Cutlery","Kitchen Gadgets","Laptop Accessories","Other","Outdoor Living","Printer Ink & Toner",
            "Ranges","Sheet Music","Speakers","TV Stands","iPhone Accessories"
        ]

    cat_level_4 = [
            "All TV Stands","Cases","Cookware","DSLR Lenses","Deck Installation Parts","Electric Ranges",
            "Food Preparation Utensils","Gas Ranges","Laptop Bags & Cases","Other","PC Laptops","Patio Furniture & Decor",
            "Portable Chargers/Power Packs","Printer Ink","Smartwatch Accessories","Toner","iPhone Cases & Clips"
        ]

    cat_level_5 = [
            "Antennas & Adapters","Coffee Pods","Condenser","DSLR Flashes","Dart Board Cabinets","Dash Installation Kits",
            "Deck Harnesses","Drink & Soda Mixes","Electric Espresso Machines","Electric Tea Kettles","Fire Pits",
            "Flash Accessories","Full-Size Blenders","In-Ceiling Speakers","In-Wall Speakers","Inkjet Printers",
            "Interfaces & Converters","Laser Printers","Multi-Cup Coffee Makers","Other","Ottomans","Outdoor Furniture Sets",
            "Outdoor Seating","Prime Lenses","Single-Serve Blenders","Single-Serve Coffee Makers","Smartwatch Bands",
            "Universal Camera Bags & Cases","Wireless & Bluetooth Mice","iPhone 6s Cases","iPhone 6s Plus Cases"
        ]
    return cat_level_1, cat_level_2, cat_level_3, cat_level_4, cat_level_5


def process_predictions(pred_1: list, pred_2: list, pred_3: list, pred_4:list, pred_5: list):
    """
    Process prediction arrays to find the corresponding category names and subcategories.
    
    Args:
        pred_1, pred_2, pred_3, pred_4, pred_5 (list): Lists of prediction scores.
        
    Returns:
        tuple: A tuple containing the category names and corresponding subcategory names.
    """
    # All categories
    categories = pd.read_json("categories/categories.json") 
    # Categories
    cat_level_1, cat_level_2, cat_level_3, cat_level_4, cat_level_5 = _get_categories()

    # Get the maximum indices for predictions
    max_index_1 = np.argmax(pred_1)
    max_index_2 = np.argmax(pred_2)
    max_index_3 = np.argmax(pred_3)
    max_index_4 = np.argmax(pred_4)
    max_index_5 = np.argmax(pred_5)

    # Get the names of the corresponding categories
    category_name_1 = cat_level_1[max_index_1]
    category_name_2 = cat_level_2[max_index_2]
    category_name_3 = cat_level_3[max_index_3]
    category_name_4 = cat_level_4[max_index_4]
    category_name_5 = cat_level_5[max_index_5]
    
    # Validate the categories
    if not is_category_valid(category_name_1, categories):
        category_name_1 = 'Other'

    result2 = None
    result3 = None
    result4 = None
    result5 = None

    if not is_category_valid(category_name_2, categories):
        result2 = 'Other'
    else:
        if not is_subcategory_valid(category_name_1, category_name_2, categories):
            result2 = 'Other'
        else:
            result2 = category_name_2

    if result2 != 'Other' and not is_category_valid(category_name_3, categories):
        result3 = 'Other'
    else:
        if not is_subcategory_valid(category_name_2, category_name_3, categories):
            result3 = 'Other'
        else:
            result3 = category_name_3

    if result3 != 'Other' and not is_category_valid(category_name_4, categories):
        result4 = 'Other'
    else:
        if not is_subcategory_valid(category_name_3, category_name_4, categories):
            result4 = 'Other'
        else:
            result4 = category_name_4

    if result4 != 'Other' and not is_category_valid(category_name_5, categories):
        result5 = 'Other'
    else:
        if not is_subcategory_valid(category_name_4, category_name_5, categories):
            result5 = 'Other'
        else:
            result5 = category_name_5
    return category_name_1, result2, result3, result4, result5
