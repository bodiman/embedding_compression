"""
    Takes a list of strings and splits strings into strings of a maximum size

    Parameters
    ----------

    text_batch: str[] 
        Batch of text input

    buffer_size: int
        Maximum number of characters to be included in new strings

    Returns
    -------
    
    low_token_batch: str

"""
def buffer_text(text_batch, buffer_size):
    buffer_list = []

    for text in text_batch:
        for i in range(0, len(text), buffer_size):
            buffer_list.append(text[i:i+buffer_size])

    return buffer_list