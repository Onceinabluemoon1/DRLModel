"""Back-compat re-exports; prefer ``encoders`` and ``decoders`` directly."""

from pref_learn.models.decoders import (  # noqa: F401
    DECODER_TYPES,
    SUM_HEAD_DECODER_TYPES,
    DecoderType,
    MultiplicativeDecoder,
    ConditionalDecoder,
    SlotDecoder,
    OdinDecoder,
    RewardDecoderConfig,
    decoder_sums_latent_heads,
    make_reward_decoder,
    normalize_decoder_type,
    reward_decoder_column_orth_penalty,
    reward_decoder_theta_sparsity,
    split_obs_z,
)
from pref_learn.models.encoders import (  # noqa: F401
    PairEncoder,
    PairEncoderOutput,
)
