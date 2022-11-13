def average_word_len(string):
    """Function that returns average word length

    Args:
        string (str): A text string.
    
    Returns:
        int: Number of words.
    """
    
    # Split the string into words.
    words = string.split()
    
    # Compute length of each word and store in a separate list.
    words_len = [len(word) for word in words]
    
    # Return average word length
    return sum(words_len)/len(words)
    
    