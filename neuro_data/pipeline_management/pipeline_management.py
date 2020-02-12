import datajoint as dj

from neuro_data.static_images.data_schemas import StaticScanCandidate, StaticScan, ConditionTier, Frame, InputResponse, Eye, Treadmill, StaticMultiDataset, StaticMultiDatasetGroupAssignment, ExcludedTrial

pipeline_anatomy = dj.create_virtual_module('pipeline_anatomy', 'pipeline_anatomy')
pipeline_fuse = dj.create_virtual_module('pipeline_fuse', 'pipeline_fuse')
pipeline_stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')


class NeuroDataPipelineManagement():
    def __init__(self):
        pass

    def manually_insert_area_for_scan(self, target_scan, area):
        """
        Give a target_scan, it will label all neurons in there with the given area

        Parameters:
        target_scan : dict(animal_id, session, scan_idx)
        area : str

        Returns:
        None
        """
        neuron_unit_keys = (pipeline_fuse.ScanSet().Unit() & target_scan).fetch('KEY')
        for neuron_unit_key in neuron_unit_keys:
            neuron_unit_key['brain_area'] = area
            pipeline_anatomy.AreaMembership().insert1(neuron_unit_key, allow_direct_insert=True)

    def manually_insert_layer_for_scan(self, target_scan, layer):
        """
        Give a target_scan, it will label all neurons in there with the given layer

        Parameters:
        target_scan : dict(animal_id, session, scan_idx)
        layer : str

        Returns:
        None
        """
        neuron_unit_keys = (pipeline_fuse.ScanSet().Unit() & target_scan).fetch('KEY')
        for neuron_unit_key in neuron_unit_keys:
            neuron_unit_key['layer'] = layer
            pipeline_anatomy.LayerMembership().insert1(neuron_unit_key, allow_direct_insert=True)

    # populate functions
    def process_static_scan(self, target_scan):
        """
        Function that goes and check for every table that needs to be populate as well as provide an option
        to manaully populate AreaMembership and LayerMembership, assuming that all the neurons can be label the same Area and Layer
        if not, they the user should manually do it.

        Please refer to neuro_data/notebooks/pipeline_management notebook for an example

        Parameters:
        target_scan : dict(animal_id, session, scan_idx)

        Returns:
        None
        """
        print('Running preprocessing checks for ', target_scan)

        # Check if the scan has been processed completely
        if not pipeline_fuse.ScanDone() & target_scan:
            print('[Preprocessing Check]: ' + str(target_scan) + ' Scan has not been processed yet, please look into pipeline for details')
            return
        else:
            print('[Preprocessing Check]: ScanDone Check Passed')

        # Check if neurons area are labeled
        if not pipeline_anatomy.AreaMembership() & target_scan:
            print('[Preprocessing Check]: ' + str(target_scan) + " AreaMembership is not populated")
            user_input = None
            while user_input not in ['y', 'n']:
                user_input = input('Should we manually insert if the area are known and all the same? [y/n]')
                
                if user_input == 'y':
                    area = input('Input area to label neurons with [type exit for cancel]:')
                    while area not in ['V1', 'LM', 'AL', 'RL', 'all-unknown']:
                        if area == 'exit':
                            return
                        print('Invalid Area!')
                        area = input('Input area to label neurons with:')
                        
                    self.manually_insert_area_for_scan(target_scan, area)
                elif user_input == 'n':
                    return
        else:
            print('[Preprocessing Check]: AreaMembership Check Passed')

        # Check if neuron layers are labeled
        if not pipeline_anatomy.LayerMembership() & target_scan:
            print('[Preprocessing Check]: ' + str(target_scan) + " LayerMembership is not populated")

            user_input = None
            while user_input not in ['y', 'n']:
                user_input = input('Should we manually insert if the layer are known and all the same? [y/n]')
                
                if user_input == 'y':
                    layer = input('Input layer to label neurons with [type exit to cancel]:')
                    while layer not in ['L1', 'L2/3', 'L4']:
                        if layer == 'exit':
                            return
                        print('Invalid Layer!')
                        layer = input('Input layer to label neurons with:')
                        
                    
                    self.manually_insert_layer_for_scan(target_scan, layer)
                elif user_input == 'n':
                    return
        else:
            print('[Preprocessing Check]: LayerMembership Check Passed')

        # Check pipeline_stimulus.Sync() table
        if not pipeline_stimulus.Sync() & target_scan:
            print('[Preprocessing Check]: ' + str(target_scan) + ' pipeline_stimulus.Sync() table is not processed or failed to processed')
            return
        else:
            print('[Preprocessing Check]: Sync Check Passed')

        # All tables requirements are met, begin neurodata dataset population
        print('[Preprocessing Check]: All table requirements passed, beginning neuro_data populating:')
        
        # Get the ScanDone primary key reference
        target_scan_done_key = (pipeline_fuse.ScanDone() & target_scan).fetch1('KEY')

        # Insert into StaticScanCandidate
        if not StaticScanCandidate & target_scan_done_key:
            StaticScanCandidate.insert1(target_scan_done_key)
            print('[NeuroData.Static Populate]: Successfully inserted Scan into StaticScanCandidate')
        else:
            print('[NeuroData.Static Populate]: Scan has already been added to StaticScanCandidate')

        # Populating StaticScans
        print("[NeuroData.Static Populate]: Populating StaticScan:")
        StaticScan().populate(target_scan_done_key)

        # Populate ConditionTier
        print("[NeuroData.Static Populate]: Populating ConditionTier:")
        ConditionTier.populate(target_scan_done_key)

        # Check for incorrect flip times
        print("[NeuroData.Static Populate]: Checking for Incorrect Flip Times:")
        trials = (pipeline_stimulus.Trial() & target_scan).proj('flip_times').fetch(as_dict=True)
        for trial in trials:
            if trial['flip_times'].shape[1] != 3: # correct number of flips, hardcoded
                ExcludedTrial.insert1(trial, ignore_extra_fields=True)

        # Populate Frame
        print("[NeuroData.Static Populate]: Populating Frame:")
        Frame.populate(dict(preproc_id = 0))

        # Populate InputResponse
        print("[NeuroData.Static Populate]: Populating InputResponse:")
        InputResponse().populate(target_scan_done_key, dict(preproc_id = 0))

        # Populate Eye
        print("[NeuroData.Static Populate]: Populating Eye:")
        Eye().populate(target_scan_done_key)

        # Populate Treadmill
        print("[NeuroData.Static Populate]: Populating Treadmill:")
        Treadmill().populate(target_scan_done_key)

        # Insert Scan into StaticMultiDatasetGroupAssignment with whatever is the next highest_group_id
        print("[NeuroData.Static Populate]: Inserting Scan into StaticMultiDatasetGroupAssignment with next largest group_id:")
        target_input_response_key = (InputResponse & target_scan & dict(preproc_id=0)).fetch1('KEY')
        if not StaticMultiDatasetGroupAssignment & target_input_response_key:
            target_input_response_key['group_id'] = StaticMultiDatasetGroupAssignment().fetch('group_id').max() + 1
            target_input_response_key['description'] = 'Inserted from PipelineManagement'
            StaticMultiDatasetGroupAssignment.insert1(target_input_response_key)
        else:
            print("[NeuroData.Static Populate]: Scan is already in StaticMultiDatasetGroupAssignment, skipping")

        # Fill StaticMultiDataset
        print("[NeuroData.Static Populate]: Filling StaticMultiDataset:")
        StaticMultiDataset().fill()

        print('[PROCESSING COMPLETED FOR SCAN: ' + str(target_scan) + ']\n')