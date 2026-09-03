"""Hand-written Triton kernels for the optimized Transformer layer."""

from .fused_layernorm import (  # noqa: F401
    HAVE_TRITON,
    HAVE_TRITON_OP,
    MAX_FUSED_WIDTH,
    can_fuse,
    fused_add_layernorm,
    fused_layernorm_module,
)

from .attention import (  # noqa: F401
    HAVE_ATTN_OP,
    MAX_HEAD_DIM,
    attention as triton_attention,
    attention_raw,
    can_use as can_use_attention,
)

from .fp16x3 import (  # noqa: F401
    HAVE_X3_TRITON_OP,
    HAVE_X3_LINEAR_OP,
    MAX_X3_WIDTH,
    x3_available,
    x3_selfcheck,
    x3_can_use,
    x3_prepare,
    x3_linear,
    x3_ln_split,
    x3_ln,
    x3_add_ln_split,
    x3_act_split,
)
