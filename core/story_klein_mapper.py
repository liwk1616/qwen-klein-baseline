"""A learned visual-output bottleneck mapping frozen Qwen context to Klein."""
import torch
from torch import nn


MAPPER_SYSTEM = ('Read the preceding Flintstones images and their captions in chronological order. '
    'Represent the next requested frame, resolving person references and preserving the requested event and location. '
    'Earlier images may be generated and contain mistakes; the current caption determines who acts and what changes. '
    'Do not add people or events absent from the current description. No future frame is supplied.')


def mapper_messages(history, current_caption):
    content = []
    for previous in history:
        content.extend([{'type': 'image'}, {'type': 'text', 'text': 'Previous frame caption: ' + previous['caption']}])
    content.append({'type': 'text', 'text': 'Current requested frame: ' + current_caption + '\nRepresent this frame for image generation.'})
    return [{'role': 'system', 'content': MAPPER_SYSTEM}, {'role': 'user', 'content': content}]


class StoryConditionMapper(nn.Module):
    """Eight learned output queries, then native-length conditional decoding.

    Unlike the original paper's learned tokens inside OPT/Llama, these queries
    attend to frozen Qwen states. This preserves cached feature training and is
    an explicit Klein/Qwen adaptation, not a claim of identical implementation.
    """
    def __init__(self, hidden=512, text_dim=7680, vision_language_dim=2560, length=128):
        super().__init__()
        self.state_norm = nn.LayerNorm(vision_language_dim)
        self.state_projection = nn.Linear(vision_language_dim, hidden)
        self.output_queries = nn.Parameter(torch.randn(1, 8, hidden) * .02)
        self.condition_queries = nn.Parameter(torch.randn(1, length, hidden) * .02)
        layer = nn.TransformerDecoderLayer(hidden, 8, hidden*4, dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.output_decoder = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(hidden))
        self.condition_decoder = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(hidden))
        self.current_norm = nn.LayerNorm(text_dim)
        self.current_projection = nn.Linear(text_dim, hidden)
        self.delta = nn.Linear(hidden, text_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, qwen_states, current_condition):
        if qwen_states.shape[1:] != (8, self.state_norm.normalized_shape[0]):
            raise ValueError('Expected eight causal Qwen boundary states')
        if current_condition.shape[1:] != (self.condition_queries.shape[1], self.current_norm.normalized_shape[0]):
            raise ValueError('Native conditional sequence differs from mapper configuration')
        state = self.state_projection(self.state_norm(qwen_states.float()))
        queries = self.output_queries.expand(qwen_states.shape[0], -1, -1)
        visual_output = self.output_decoder(queries, state)
        current = self.current_projection(self.current_norm(current_condition.float()))
        condition = self.condition_decoder(self.condition_queries + current, visual_output)
        return current_condition + self.delta(condition).to(current_condition.dtype)
