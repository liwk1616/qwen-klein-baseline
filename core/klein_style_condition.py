"""One dataset-level style instruction, identical in training and inference."""
STYLE_SUFFIX = '\nA frame from The Flintstones animated cartoon.'


def klein_prompt(caption):
    return caption.strip() + STYLE_SUFFIX
