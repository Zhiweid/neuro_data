import datajoint as dj
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
from scipy.stats import pearsonr
from foundation.fnn.data import Data
from foundation.fnn.model import Model, Instance
from foundation.fnn.train import Objective, Train
from foundation.utility.resize import Resize
from foundation.stimulus.video import FrameList
from foundation.recording.scan import ScanUnitOrder
from foundation.recording.trace import Trace
from foundation.fnn.visual import VisualRecordingCorrelation
from foundation.recording.visual import VisualMeasure
from neuro_data import logger as log
from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images import data_schemas
from neuro_data.static_images.data_schemas import Preprocessing, SplineCurve, FilterMixin, stimulus, fuse

stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

schema = dj.schema('neurodata_foundation_static')

@schema
class FoundationInputResponse(dj.Computed, FilterMixin):
    definition = """ # foundation model predicted responses of all neurons to all images
    -> Model
    -> FrameList
    -> Preprocessing
    ---
    """
    
    class Input(dj.Part):
        definition = """
        -> master
        -> stimulus.Frame
        ---
        row_id              : int         # row id in the response block
        -> stimulus.StaticImage.Image
        """

    class ResponseBlock(dj.Part):
        definition = """
        -> master
        ---
        responses           : blob@data   # response of all neurons to all trials in shape of (n_frames, n_neurons)
        """

    class ResponseKeys(dj.Part):
        definition = """
        -> master
        -> fuse.Activity.Trace
        ---
        col_id              : int         # col id in the response block
        """
    
    @property
    def key_source(self):
        return Model * FrameList * Preprocessing & 'preproc_id = 8'
    
    def make(self, key):
        # Get stimulus parameters used for foundation model training
        model = (Model & key).model(device="cuda")
        model_period = (Data & key).link.compute.sampling_period
        model_offset = (Data & key).link.compute.unit_offset
        height, width = (Data & key).link.compute.resolution
        resize_id = (Data & key).link.compute.resize_id
        burnin_frames = ((Objective & (Train & (Instance() & key).link).link).link).fetch1('burnin_frames')
        
        # Create stimulus generator
        video = (FrameList() & key).compute.video
        rvideo = (Resize & {"resize_id": resize_id}).link.resize(video=video, height=height, width=width)
        stimuli = rvideo.generate(period=model_period)

        # And we'll feed that to our model, collecting responses in a list
        traces = []
        for r in model.generate_response(stimuli=stimuli):
            traces.append(r)
        traces = np.stack(traces, axis=0)

        # Since we have the number of frames, the sampling period and offset, we know the timing of the response
        frame_times = np.arange(traces.shape[0]) * model_period + model_offset
        
        # Integration window size for the predicted trace
        duration, offset = map(float, (Preprocessing & key).fetch1('duration', 'offset'))
        filter_type = (Preprocessing & key).fetch1('filter')
        sample_point = offset + duration / 2

        log.info('Generating lowpass filters to {}Hz'.format(1 / duration))
        downsample_filter = self.get_filter(duration, model_period, filter_type, warning=False)
        
        # Low pass filter the traces after removing the burnin frames
        R = [] 
        for trace in traces.T:
            trace_spline = SplineCurve(frame_times[burnin_frames:],
                                       [np.convolve(trace[burnin_frames:], downsample_filter, mode='same')], k=1, ext=1)
        
            # Compute onset time of each trial (i.e. when pre_blank ends and the image starts), see details of how video.times is computed  
            # at foundation.stimulus.video.FrameList.compute
            stimulus_onset = np.array(video.times[1:])[::2]
            if stimulus_onset[0] < frame_times[burnin_frames]:
                warnings.warn('First trial onset is within the burn-in period!')
            
            # Get interpolated trial responses
            _R = trace_spline(stimulus_onset + sample_point, log=False)
            R.append(_R.squeeze())
        R = np.stack(R).T
        
        # Get info of input images and neurons
        input_tups = (FrameList.Member * stimulus.Frame & key).fetch('condition_hash', 'image_class', 'image_id', order_by='framelist_index ASC', as_dict=True)
        unit_tups = (fuse.Activity.Trace * Trace.ScanUnit * (ScanUnitOrder() & (Data & key).link)).fetch(as_dict=True, order_by='unit_id ASC')

        # Re-order responses by ascending unit_id
        order = np.array([tup['trace_order'] for tup in unit_tups])
        R = R[:, order]
        
        self.insert1(key)
        self.ResponseBlock.insert1(dict(**key, responses=R))
        self.ResponseKeys.insert([dict(**key, **tup, col_id=cid) for cid, tup in enumerate(unit_tups)], ignore_extra_fields=True)
        self.Input.insert([dict(**key, **tup, row_id=rid) for rid, tup in enumerate(input_tups)])

@schema
class FoundationEval(dj.Computed):
    definition = """
    -> Model
    ---
    median_dyn_cc_abs            : float # median trial-average test correlation coefficient computed on dynamic oracles 
    median_dyn_cc_max            : float # median maximum possible correlation coefficient computed on dynamic oracles
    median_dyn_cc_norm           : float # median dyn_cc_abs / dyn_cc_max
    median_sta_cc_abs            : float # median trial-average test correlation coefficient computed on static oracles 
    median_sta_cc_max            : float # median maximum possible correlation coefficient computed on static oracles
    median_sta_cc_norm           : float # median sta_cc_abs / sta_cc_max
    """
                
    class UnitDynamic(dj.Part):
        definition = """
        -> master
        -> VisualRecordingCorrelation.proj(trace_order='unit')
        -> VisualMeasure
        unit_id                  : int   # unit_id as in fuse.Activity.Trace
        ---
        dyn_cc_abs               : float # trial-average test correlation coefficient computed on dynamic oracles 
        dyn_cc_max               : float # maximum possible correlation coefficient computed on dynamic oracles
        dyn_cc_norm              : float # dyn_cc_abs / dyn_cc_max
        """
        
    class UnitStatic(dj.Part):
        definition = """
        -> master
        -> FoundationInputResponse.ResponseKeys
        ---
        sta_cc_abs               : float # trial-average test correlation coefficient computed on static oracles 
        sta_cc_max               : float # maximum possible correlation coefficient computed on static oracles
        sta_cc_norm              : float # sta_cc_abs / sta_cc_max
        """
        
    def make(self, key):
        # Fetch dynamic cc_abs and cc_max
        trace_rel = dj.U('trace_id', 'unit_id', 'trace_order') & (Trace.ScanUnit * ScanUnitOrder * Data.VisualScan & key)
        cc_abs_rel = VisualRecordingCorrelation.proj('correlation', trace_order='unit') * trace_rel & key
        cc_max_rel = VisualMeasure * trace_rel
        assert (len(cc_abs_rel) == len(cc_max_rel)), 'number of units disagree!'
        dyn_abs_key, dyn_cc_abs = cc_abs_rel.fetch(dj.key, 'correlation', order_by='unit_id')
        dyn_max_key, dyn_cc_max = cc_max_rel.fetch(dj.key, 'measure', order_by='unit_id')

        # Replace nan value with 0.0
        dyn_cc_norm = dyn_cc_abs / dyn_cc_max
        dyn_cc_abs = np.nan_to_num(dyn_cc_abs,nan=0.0)
        dyn_cc_max = np.nan_to_num(dyn_cc_max,nan=0.0)
        dyn_cc_norm = np.nan_to_num(dyn_cc_norm,nan=0.0)
        dyn_tuples = [dict(**abs_key, resample_id=max_key['resample_id'], offset_id=max_key['offset_id'], rate_id=max_key['rate_id'], measure_id=max_key['measure_id'],
                           dyn_cc_abs=cc_abs, dyn_cc_max=cc_max, dyn_cc_norm=cc_norm)
                           for abs_key, max_key, cc_abs, cc_max, cc_norm in zip(dyn_abs_key, dyn_max_key, dyn_cc_abs, dyn_cc_max, dyn_cc_norm)]

        # Compute static cc_max 
        data_config = DataConfig.CorrectedAreaLayer & \
                         {'stimulus_type': 'stimulus.Frame', 'exclude': '', 'layer': 'L2/3',
                          'normalize_per_image': False, 'normalize': True} & 'brain_area in ("V1")'
        group = data_schemas.StaticMultiDatasetGroupAssignment & (Data.VisualScan & key) & 'preproc_id = 14'
        dset_key = (DataConfig * data_schemas.StaticMultiDataset & (group * data_config).proj()).fetch1(dj.key)
        testsets, _ = DataConfig().load_data(dset_key, tier='test', oracle=True)
        ro_key = list(testsets.keys())[0]
        # group in vivo static responses by condition_hash in the dynamic scan
        conds, im_classes, im_ids = (stimulus.Condition * stimulus.Frame * data_schemas.ConditionTier & (Data.VisualScan & key)).fetch('condition_hash', 'image_class', 'image_id', order_by='image_class, image_id')
        responses = []
        for im_c, im_id in zip(im_classes, im_ids):
            class_str = np.array([c.astype('str') for c in testsets[ro_key].item_info['frame_image_class']])
            idxs = np.where((class_str == im_c) & \
                   (np.array(list(testsets[ro_key].item_info['frame_image_id'])) == im_id))[0]
            responses.append(testsets[ro_key].responses[idxs])
        sta_cc_max = cal_reliability(responses)

        # Compute static cc_abs
        testsets, _ = DataConfig().load_data(dset_key, tier='test')
        norm_resps = testsets[ro_key].responses / testsets[ro_key].statistics['responses/all/std']
        avg_resps = []
        for im_c, im_id in zip(im_classes, im_ids):
            class_str = np.array([c.astype('str') for c in testsets[ro_key].item_info['frame_image_class']])
            idxs = np.where((class_str == im_c) & \
                   (np.array(list(testsets[ro_key].item_info['frame_image_id'])) == im_id))[0]
            avg_resps.append(norm_resps[idxs].mean(0))
        avg_resps = np.stack(avg_resps)
        framelist_key = (dj.U('framelist_id', 'preproc_id') & (FoundationInputResponse.Input & key & [{'image_class': im_c, 'image_id': im_d} for im_c, im_d in zip(im_classes, im_ids)])).fetch1()
        unit_keys = (FoundationInputResponse.ResponseKeys & framelist_key & key).fetch(dj.key, order_by='unit_id')
        rows = (FoundationInputResponse.Input & key & framelist_key & [{'image_class': im_c, 'image_id': im_d} for im_c, im_d in zip(im_classes, im_ids)]).fetch('row_id', order_by='image_id')
        resps = (FoundationInputResponse.ResponseBlock & key & framelist_key).fetch1('responses')
        resps = resps[rows]
        sta_cc_abs = np.array([pearsonr(avg_resps[:, i], resps[:, i])[0] for i in range(resps.shape[1])])

        # Replace nan value with 0.0
        sta_cc_norm = sta_cc_abs / sta_cc_max
        sta_cc_abs = np.nan_to_num(sta_cc_abs,nan=0.0)
        sta_cc_max = np.nan_to_num(sta_cc_max,nan=0.0)
        sta_cc_norm = np.nan_to_num(sta_cc_norm,nan=0.0)

        sta_tuples = [dict(**uk, sta_cc_abs=cc_abs, sta_cc_max=cc_max, sta_cc_norm=cc_norm)
                           for uk, cc_abs, cc_max, cc_norm in zip(unit_keys, sta_cc_abs, sta_cc_max, sta_cc_norm)]

        # Insert
        self.insert1(dict(**key, median_dyn_cc_abs=np.nanmedian(dyn_cc_abs), median_dyn_cc_max=np.nanmedian(dyn_cc_max), median_dyn_cc_norm=np.nanmedian(dyn_cc_norm), \
                                 median_sta_cc_abs=np.nanmedian(sta_cc_abs), median_sta_cc_max=np.nanmedian(sta_cc_max), median_sta_cc_norm=np.nanmedian(sta_cc_norm)))
        self.UnitDynamic.insert(dyn_tuples)
        self.UnitStatic.insert(sta_tuples)
        

# responses is a list of [n_repeats,n_units]
def cal_reliability(responses): 
    # Fill missing trial with NaNs for convenience
    agg_ns = np.array([len(r) for r in responses])
    max_n = max(agg_ns)
    for i, r in enumerate(responses):
        add_shape = list(r.shape)
        if add_shape[0] < max_n:
            add_shape[0] = max_n - add_shape[0]
            nan = np.full_like(r, np.nan, shape=add_shape)
            responses[i] = np.concatenate([r, nan])
    v = 1 / agg_ns**2
    w = agg_ns - 1
    z = agg_ns.sum() - len(agg_ns)
    n = np.sqrt(z.sum() / (w * v).sum())
    y = np.stack(responses, axis=0)  

    # Mean for each stimuli
    y_m = np.nanmean(y,axis=1)

    # Power: variance of mean of stimuli
    P = np.var(y_m,axis=0,ddof=1)

    # Total power: mean of variance across repeats
    TP = np.mean(np.nanvar(y, axis=0, ddof=1), axis=0)

    # Signal power: 
    SP = (n * P - TP) / (n - 1)
    # variance of response mean
    y_m_v = np.var(y_m, axis=0, ddof=0)

    # correlation coefficient ceiling
    cc_max = np.sqrt(SP / y_m_v)
    return cc_max