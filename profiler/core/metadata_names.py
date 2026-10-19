"""Descriptive acquisition policies shared by metadata producers and readers."""

PACKED_WEIGHT_POLICY = 'vllm-dummy-mxfp4-bf16-cuda-row-chunks'
MOE_WEIGHT_POLICY = 'native-rotated-expert-weights'
MOE_CONDITIONING_POLICY = 'rotated-weight-cycles'
EAGER_ATTRIBUTION = 'native-eager-launch-correlation'
GRAPH_ATTRIBUTION = 'native-graph-launch-correlation'
