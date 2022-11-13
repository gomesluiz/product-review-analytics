def count_words(string):
    """Function that returns number of words in a string.

    Args:
        string (str): A text string.
    
    Returns:
        int: Number of words.
    """
    # Split the string into words.
    words = string.split()

    # Return the number of words.
    return len(words)