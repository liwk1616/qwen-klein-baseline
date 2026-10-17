# Extracted from scripts/flintstones_klein/prepare_char_klein.py; historical orchestration removed.
import re

def caption_tokens(tokenizer, caption, length=None):
    wrapped = tokenizer.apply_chat_template([{'role': 'user', 'content': caption}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    options = {'return_offsets_mapping': True, 'truncation': False}
    if length:
        options.update(padding='max_length', max_length=length)
    encoded = tokenizer(wrapped, **options)
    return wrapped, encoded, wrapped.index(caption) if caption else 0


def segment_token_indices(caption, word, caption_start, offsets):
    matches = list(re.finditer(r'(?<!\w)' + re.escape(word) + r'(?!\w)', caption, re.IGNORECASE))
    if not matches:
        return []
    answer = []
    for match in matches:
        start, stop = caption_start + match.start(), caption_start + match.end()
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < stop and b > start]
        if indices:
            answer.append(indices[-1])
    return answer
