# Extracted from scripts/flintstones_klein/run_text_reference_factorial.py; historical orchestration removed.
from copy import deepcopy
from klein_style_condition import klein_prompt
from prepare_char_klein import caption_tokens
from story_klein_mapper import mapper_messages
from independent_candidate_protocol import name_mentions
from story_reference_refresh import caption_sources
from progressive_protocol import generator_request
PROTOCOL = "automatic_named_bank_own_history_20261004_v1"

def condition(models, request):
    torch = models.torch
    settings = request['factorial']
    prompt = klein_prompt(request['target_caption'])
    _, encoded, start = caption_tokens(models.pipe.tokenizer, prompt, 128)
    if len(encoded['input_ids']) != 128:
        raise ValueError('Changed native token budget')
    native, _ = models.pipe.encode_prompt(prompt, device=torch.device('cuda:0'), max_sequence_length=128)
    refs = token_bindings(settings['semantic_spans'], start, encoded['offset_mapping'], models.bank)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        value = native
        if settings['mapper']:
            history = settings['actual_generator_history']
            inputs = models.qwen_inputs([mapper_messages(history, request['target_caption'])], [h['image'] for h in history])
            if not bool(inputs['attention_mask'][:, -8:].all()):
                raise ValueError('Invalid Qwen causal boundary states')
            states = models.qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -8:].contiguous()
            value = models.mapper(states, native)
        if refs:
            visual = torch.stack([torch.stack([models.bank_features[m['feature_row']] for m in ref['members']]).float().mean(0).to(torch.bfloat16) for ref in refs]).unsqueeze(0)
            indices = torch.tensor([[r['text_token_index'] for r in refs]], device='cuda:0', dtype=torch.int64)
            delta = models.fusion(native, visual, indices)-native
            value = value+delta
        else:
            delta = torch.zeros_like(native)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite generation condition')
    stats = {'native_shape': list(native.shape), 'condition_delta_abs_mean': float((value-native).float().abs().mean()),
             'reference_delta_abs_mean': float(delta.float().abs().mean()),
             'plural_pooled_tokens': sum(len(r['members'])>1 for r in refs)}
    return value, prompt, refs, stats

def token_bindings(spans, caption_start, offsets, bank):
    records = []
    for span in spans:
        a, b = caption_start+span['start'], caption_start+span['end']
        hits = [i for i, (x, y) in enumerate(offsets) if y>x and x<b and y>a]
        if not hits:
            raise ValueError('Reviewed identity span has no actual native token')
        members = []
        for name in span['names']:
            if name not in bank:
                raise ValueError('Unknown identity cannot acquire a bank feature')
            index, image = bank[name]
            members.append({'name': name, 'feature_row': index, 'image': image})
        records.append({**span, 'text_token_index': hits[-1], 'members': members,
                        'aggregation': 'mean_visual_features' if len(members)>1 else 'single_feature'})
    if len({r['text_token_index'] for r in records}) != len(records):
        raise ValueError('Distinct semantic spans collide at one native token')
    return records

def automatic_request(models, request, target):
    generated = generator_request(request,target,True)
    fallback = len(caption_tokens(models.pipe.tokenizer,klein_prompt(generated['target_caption'].strip()))[1]['input_ids'])>128
    if fallback:generated = generator_request(request,target,False)
    generated['target_caption'] = generated['target_caption'].strip()
    generated['context']['current_caption'] = generated['target_caption']
    spans = [{'start':m['start'],'end':m['end'],'names':[m['name']],'text':m['text'],'source':'automatic_current_caption_name'}
             for m in name_mentions(generated['target_caption'])]
    if any(name not in caption_sources(request) for span in spans for name in span['names']):
        raise ValueError('Automatic reference identity has no name in the allowed current/past captions')
    later = request['target_position']>1
    generated['factorial'] = {'mapper':later,'reference':later,'semantic_spans':spans if later else [],
        'actual_generator_history':deepcopy(request['context']['history']) if later else []}
    generated['intervention'] = {'protocol':PROTOCOL,'reviewed_answers_used':False,'caption_length_fallback':fallback,
        'bank_identity_source':'Explicit named mentions in this frame automatic generation caption'}
    return generated
