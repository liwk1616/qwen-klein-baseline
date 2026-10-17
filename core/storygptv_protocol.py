"""Pure input protocol for StoryGPT-V's released Flintstones story evaluator.

This defines captions and allowed conditioning, not the StoryGPT-V model.
The caller must separately authorize the fixed reference-bank benchmark policy.
"""
from __future__ import annotations

CANONICAL_NAMES = ('fred', 'wilma', 'barney', 'betty', 'pebbles', 'mr slate', 'dino')


def official_story_ids(split_ids, followings, story_length=4):
    return [clip_id for clip_id in split_ids
            if clip_id in followings and len(followings[clip_id]) == story_length - 1]


def official_caption_records(clip_ids, annotations, characters):
    """Preserve released name-replacement and whitespace-token matching behavior.

    In particular, do not silently repair token punctuation, multiword names,
    or the upstream list-membership expressions when claiming this protocol.
    """
    female = ['wilma', 'betty', 'pebbles']
    male = ['fred', 'barney', 'mr slate', 'dino']
    result = []
    for index, clip_id in enumerate(clip_ids):
        caption = annotations[clip_id]['caption']
        ref_flag = False
        if index == 0:
            immediate = characters[clip_id]
        elif sorted(characters[clip_id]) == sorted(immediate) or characters[clip_id] in immediate:
            if len(immediate) > 1:
                if len(immediate) == 2:
                    name = immediate[0].capitalize() + ' and ' + immediate[1].capitalize()
                    if name not in caption:
                        name = immediate[1].capitalize() + ' and ' + immediate[0].capitalize()
                    caption = caption.replace(name, 'They')
                elif len(immediate) == 3:
                    name = immediate[0].capitalize() + ', ' + immediate[1].capitalize() + ' and ' + immediate[2].capitalize()
                    caption = caption.replace(name, 'They')
            elif immediate[0] in female or characters[clip_id] in female:
                caption = caption.replace(immediate[0].capitalize(), 'She')
            elif immediate[0] in male or characters[clip_id] in male:
                caption = caption.replace(immediate[0].capitalize(), 'He')
            ref_flag = True
        else:
            immediate = characters[clip_id]
        names, marked = [], []
        for token in caption.split(' '):
            marked.append(token)
            if token.lower() in CANONICAL_NAMES:
                marked.append('<|image|>')
                names.append(token.lower())
                ref_flag = False
        result.append({'clip_id': clip_id, 'caption': caption, 'ref_flag': ref_flag,
                       'marked_caption': ' '.join(marked), 'parsed_character_names': names,
                       'active_reference_bank_names': names if index == 0 else []})
    return result


def causal_history(story_id, targets, completed, next_position):
    """Only prior generated images and captions are allowed for the next frame."""
    if not 0 <= next_position < len(targets) or len(completed) != next_position:
        raise ValueError('History must contain exactly the completed prefix')
    history = []
    for position, output in enumerate(completed):
        if output['story_id'] != story_id or output['clip_id'] != targets[position]['clip_id']:
            raise ValueError('History contains another story or a reordered target')
        if output.get('status') != 'complete' or output.get('kind') != 'generated_story_frame':
            raise ValueError('History must contain generated frames, never original targets')
        history.append({'position': position, 'caption': targets[position]['caption'],
                        'image': output['image'], 'kind': 'generated_story_frame'})
    return {'story_id': story_id, 'history': history, 'current_caption': targets[next_position]['caption'],
            'current_position': next_position}
