import re

def get_mt_window(mention, mention_text, window=64):
    """
    Finds a mention using regex and returns a surrounding context window.
    """
    # re.escape handles cases where the mention itself has (+), [], etc.
    # We use a pattern that looks for the mention possibly followed by medical symbols
    pattern = re.escape(mention)

    match = re.search(pattern, mention_text)
    mention_start, mention_end = match.span()

    # Calculate the window boundaries
    start = max(0, mention_start - window)
    end = min(len(mention_text), mention_end + window)

    return mention_text[start:end]