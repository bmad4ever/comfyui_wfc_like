from .nodes import *

NODE_CLASS_MAPPINGS = {
    "WFC_SampleNode_BMad": WFC_SampleNode,
    "WFC_Generate_BMad": WFC_GenerateNode,
    "WFC_Encode_BMad": WFC_Encode,
    "WFC_Decode_BMad": WFC_Decode,
    "WFC_CustomTemperature_Bmad": WFC_CustomTemperature,
    "WFC_CustomValueWeights_Bmad": WFC_CustomValueWeights,
    "WFC_EmptyState_Bmad": WFC_EmptyState,
    "WFC_Filter_Bmad": WFC_Filter,
    "WFC_GenParallel_Bmad": WFC_GenParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WFC_SampleNode_BMad": "Sample (WFC)",
    "WFC_Generate_BMad": "Generate (WFC)",
    "WFC_Encode_BMad": "Encode (WFC)",
    "WFC_Decode_BMad": "Decode (WFC)",
    "WFC_CustomTemperature_Bmad": "Custom Temperature Config (WFC)",
    "WFC_CustomValueWeights_Bmad": "Custom Value Weights (WFC)",
    "WFC_EmptyState_Bmad": "Empty State (WFC)",
    "WFC_Filter_Bmad": "Filter (WFC)",
    "WFC_GenParallel_Bmad": "Parallel Multi Gen. (WFC)",
}