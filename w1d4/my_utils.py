import torch as t
import torch.nn.functional as F


def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.

    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.int64, device=device)
        logits = model(new_input_ids.unsqueeze(0))[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)

    return tokenizer.decode(input_ids + generated)


def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    return logits.argmax()


def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    return t.distributions.categorical.Categorical(logits=logits).sample()


def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    return logits / temperature


def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    minlength = logits.shape[-1]
    counts = t.bincount(input_ids, minlength=minlength)
    return logits - counts * freq_penalty


def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    topk_vals_idx = t.topk(logits, k=top_k)
    top_logits = t.full_like(logits, -t.inf)
    for idx, val in zip(topk_vals_idx.indices, topk_vals_idx.values):
        top_logits[idx] = val
    return t.distributions.categorical.Categorical(logits=top_logits).sample()


def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    probs = F.softmax(logits, dim=-1)
    sorted = probs.sort(descending=True)
    cumsum = sorted.values.cumsum(-1)
    num_indices = t.lt(cumsum, top_p).sum().item() + 1
    if num_indices < min_tokens_to_keep:    
        num_indices = min_tokens_to_keep
    keep_indices = sorted.indices[:num_indices]
    

    top_p_logits = t.full_like(logits, -t.inf)
    for idx in keep_indices:
        top_p_logits[idx] = logits[idx]

    return t.distributions.categorical.Categorical(logits=top_p_logits).sample()
