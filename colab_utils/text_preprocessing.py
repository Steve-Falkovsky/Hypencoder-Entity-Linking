import re

def get_mt_window(mention, mention_text, context_window):
    """
    Finds a mention using regex and returns a surrounding context window.
    
    Args:
        mention (str): The mention to find in the text.
        mention_text (str): The text in which to search for the mention.
        context_window (int): The number of characters to include before and after the mention.
        
    Returns:
        str: A substring of mention_text that includes the mention and its surrounding context.
    """
    # re.escape handles cases where the mention itself has (+), [], etc.
    # We use a pattern that looks for the mention possibly followed by medical symbols
    pattern = re.escape(mention)

    match = re.search(pattern, mention_text)
    mention_start, mention_end = match.span()

    # Calculate the window boundaries
    start = max(0, mention_start - context_window//2)
    end = min(len(mention_text), mention_end + context_window//2)

    return mention_text[start:end]