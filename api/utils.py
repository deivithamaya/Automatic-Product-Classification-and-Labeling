import hashlib
import os


def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files. This is, files with
    extension ".png", ".jpg", ".jpeg" or ".gif".

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    # Read file content and generate md5 hash
    file_hash = hashlib.md5(file.read()).hexdigest()
    # Return file pointer to the beginning
    file.seek(0)
    # Add original file extension
    file_ext = os.path.splitext(file.filename)[-1].lower()
    new_filename = f"{file_hash}{file_ext}"

    return new_filename

def get_str_categories(categories_name: dict):
    """
    Takes a dictionary with category names and converts it into a string.

    Parameters
    --------------
    categories_name: dict
        A dictionary containing category names.

    Returns
    --------------
    str
        A string with all category names concatenated.
    """
    result = ""
    for key, value in categories_name.items():
        if value == "Other" or value is None:
            break
        if result:
            result += " > "
        result += value
    
    return result