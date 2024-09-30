from .attention_layer import SmilesAttentionLayer
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder
from fast_transformers.builders.attention_builders import AttentionBuilder


from .attention_layer import RotateAttentionLayer 

N_HEAD = 8 #12#8
N_EMB = 128 #64#128

class RotateEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = RotateEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return RotateAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer

    
        
class SmilesEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = SmilesEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return SmilesAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoderPos

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayerPos
