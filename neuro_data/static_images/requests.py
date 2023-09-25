from .dataset_config import (
    InputConfig,
    ResponseConfig,
    TierConfig,
    LayerConfig,
    AreaConfig,
    StatsConfig,
)
from .data_schemas import StaticScan
from .ds_pipe import DvScanInfo, DvModelConfig
import datajoint as dj

schema = dj.schema("neurodata_static")
fnn = dj.create_virtual_module('fnn', 'foundation_fnn')

@schema
class DvScanInfoRequest(dj.Manual):
    definition = """
    -> StaticScan
    -> DvModelConfig
    """

# @schema
# class FoundationDynamicStaticNoBehRequest(dj.Manual):
#     definition = """
#     -> StaticScan.proj(dynamic_animal_iddynamic_session='session', dynamic_scan_idx='scan_idx')
#     -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
#     -> fnn.Model
#     -> InputConfig
#     -> TierConfig
#     -> LayerConfig
#     -> AreaConfig
#     -> StatsConfig
#     """

@schema
class DynamicStaticNoBehRequest(dj.Manual):
    definition = """
    -> DvScanInfo.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """

@schema
class DynamicStaticNoBehDiffAnimalRequest(dj.Manual):
    definition = """
    -> DvScanInfo.proj(dynamic_animal_id='animal_id', dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_animal_id='animal_id', static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """

@schema
class DynamicStaticNoBehAugRespRequest(dj.Manual):
    definition = """
    -> DvScanInfo.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> ResponseConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """
