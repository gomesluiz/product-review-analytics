def count_hashtags(string):
    """Function that returns numner of hashtags in a string.

    Args:
        string (str): A text string.
    
    Returns:
        int: Number of words.
    """
    # Split the string into words.
    words = string.split()

    # Create list of words that are hashtags.
    hashtags = [word for word in words if word.startswith("#")]

    return(len(hashtags))

