VOCAB_OURS = 50304
SEQ_LEN = 2048
WORLD_BATCH_SIZE = 2048.0
HEAD_SIZE = 128
EXPAND_FACTOR = 4.0


def flops_per_token_gqa(
    width: int,
    depth: int,
    vocab_size: int = VOCAB_OURS,
    queries_per_group: int = 2,
    seq_len: int = SEQ_LEN,
) -> float:
    """Compute FLOPs per token for a GQA transformer."""
    num_qheads = width / HEAD_SIZE
    num_kvheads = 2 * num_qheads / queries_per_group

    embeddings = 0
    attention = 2.0 * seq_len * (num_qheads + num_kvheads) * width * HEAD_SIZE
    attention += 3.5 * seq_len * (num_qheads + num_kvheads / 2) * HEAD_SIZE

    kq_logits = 1.0 * seq_len * seq_len * HEAD_SIZE * num_qheads
    softmax = 3.0 * seq_len * seq_len * num_qheads
    softmax_q_red = 2.0 * seq_len * seq_len * HEAD_SIZE * num_qheads
    final_linear = 2.0 * seq_len * width * HEAD_SIZE * num_qheads

    attn_bwd = (
        2.0 * attention
        + 2.5 * (kq_logits + softmax + softmax_q_red)
        + 2.0 * final_linear
    ) * depth

    attention += kq_logits + softmax + softmax_q_red + final_linear

    ffw_size = EXPAND_FACTOR * width
    dense_block = 6.0 * seq_len * width * ffw_size
    dense_block += 10 * seq_len * ffw_size
    dense_block += 2.0 * width * seq_len

    rmsnorm = 2 * 7.0 * width * seq_len
    final_rms_norm = 7.0 * width * seq_len
    final_logits = 2.0 * seq_len * width * vocab_size

    nonattn_bwd = 2.0 * (
        embeddings
        + depth * (dense_block + rmsnorm)
        + final_rms_norm
        + final_logits
    )

    forward_pass = (
        embeddings
        + depth * (attention + dense_block + rmsnorm)
        + final_rms_norm
        + final_logits
    )

    backward_pass = attn_bwd + nonattn_bwd
    return (forward_pass + backward_pass) / seq_len
