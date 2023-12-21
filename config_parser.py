import argparse
import os
import yaml
from os.path import dirname, abspath
import ruamel


class Parser:
    def __init__(self):
        self.seq_thres = None
        self.separated_channels = None
        self.exclude_modes = None
        self.get_position = None
        self.motorized = None
        self.location_transfer = None
        self.motion_transfer = None
        self.learning_rate = None
        self.location_epochs = None
        self.motion_epochs = None
        self.epochs = None
        self.D = None
        self.L = None
        self.val_oversampling = None
        self.val_bag_position = None
        self.train_bag_position = None
        self.train_oversampling = None
        self.test_oversampling = None
        self.test_bag_position = None
        self.in_bags = None
        self.batch_size = None
        self.statistical_features = None
        self.time_features = None
        self.spectro_augmentations = None
        self.spectro_overlap = None
        self.spectro_window = None
        self.log_power = None
        self.f_interp = None
        self.motion_augmentations = None
        self.combine_sensors = None
        self.motion_form = None
        self.motion_features = None
        self.train_val_hold_out = None
        self.train_val_split = None
        self.train_test_hold_out = None
        self.train_test_split = None
        self.diff_threshold = None
        self.inertial_positions = None
        self.gps_position = None
        self.mot_bag_step = None
        self.mot_bag_size = None
        self.mot_stride = None
        self.mot_length = None
        self.loc_stride = None
        self.loc_length = None
        self.loc_rescaler = None
        self.mot_virtuals = None
        self.loc_virtuals = None
        self.mot_rescaler = None
        self.mot_virtual = None
        self.loc_virtual = None
        self.mot_fil_w = None
        self.loc_fil_w = None
        self.mot_filter = None
        self.loc_filter = None
        self.mot_cleaner = None
        self.loc_cleaner = None
        self.old_motion_fs = None
        self.motion_fs = None
        self.old_location_T = None
        self.location_T = None
        self.max_distance = None
        self.sampling_method = None
        self.new_fs = None
        self.old_fs = None
        self.signals = None
        self.dataset = None
        self.path = None
        self.parser = argparse.ArgumentParser(
            description="pre-processing and training parameters"
        )

    def __call__(self, *args, **kwargs):
        project_root = dirname(abspath(__file__))
        config_path = os.path.join(project_root, 'config.yaml')

        self.parser.add_argument(
            '--config',
            default=config_path,
            help='config file location'
        )

        self.parser.add_argument(
            '--extract_args',
            default=dict(),
            type=dict,
            help='pre-processing arguments'
        )

        self.parser.add_argument(
            '--resample_args',
            default=dict(),
            type=dict,
            help='pre-processing arguments'
        )

        self.parser.add_argument(
            '--modify_args',
            default=dict(),
            type=dict,
            help='pre-processing arguments'
        )

        self.parser.add_argument(
            '--segment_args',
            default=dict(),
            type=dict,
            help='pre-processing arguments'
        )
        
        self.parser.add_argument(
            '--build_args',
            default=dict(),
            type=dict,
            help='building arguments'
        )

        self.parser.add_argument(
            '--train_args',
            default=dict(),
            type=dict,
            help='training arguments'
        )
        
        self.parser.add_argument(
            '--postprocess_args',
            default=dict(),
            type=dict,
            help='postprocessing arguments'
        )

    def get_args(self):
        self.__call__()
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            defaultArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for defaultKey in defaultArgs.keys():
            if defaultKey not in keys:
                print('WRONG ARG: {}'.format(defaultKey))
                assert (defaultKey in keys)

        self.parser.set_defaults(**defaultArgs)
        args = self.parser.parse_args(args=[])

        self.path = args.extract_args['path']
        self.dataset = args.extract_args['dataset']
        self.signals = args.extract_args['signals']

        self.old_motion_fs = args.resample_args['old_motion_fs']
        self.motion_fs = args.resample_args['new_motion_fs']
        self.old_location_T = args.resample_args['old_location_T']
        self.location_T = args.resample_args['new_location_T']
        self.max_distance = args.resample_args['max_distance']
        self.sampling_method = args.resample_args['sampling_method']

        self.loc_filter = args.modify_args['loc_filter']
        self.mot_filter = args.modify_args['mot_filter']
        self.mot_smoother = args.modify_args['mot_smoother']
        self.mot_sm_w = args.modify_args['mot_smoother_window']
        self.loc_fil_w = args.modify_args['loc_filter_window']
        self.mot_fil_w = args.modify_args['mot_filter_window']
        self.loc_virtuals = args.modify_args['loc_virtual_features']
        self.mot_virtuals = args.modify_args['mot_virtual_signals']
        self.mot_rescaler = args.modify_args['mot_rescaler']
        self.loc_rescaler = args.modify_args['loc_rescaler']

        self.loc_length = args.segment_args['loc_length']
        self.loc_stride = args.segment_args['loc_stride']
        self.mot_length = args.segment_args['mot_length']
        self.mot_stride = args.segment_args['mot_stride']
        self.mot_bag_size = args.segment_args['mot_bag_size']
        self.mot_bag_step = args.segment_args['mot_bag_step']
        
        self.gps_position = args.build_args['GPS_position']
        self.inertial_positions = args.build_args['inertial_positions']
        self.diff_threshold = args.build_args['diff_threshold']
        self.train_test_split = args.build_args['train_test_split']
        self.train_test_hold_out = args.build_args['train_test_hold_out']
        self.train_val_split = args.build_args['train_val_split']
        self.train_val_hold_out = args.build_args['train_val_hold_out']

        self.in_bags = args.build_args['in_bags']
        self.get_position = args.build_args['get_position']
        self.motion_features = args.build_args['motion_features']
        self.motion_form = args.build_args['motion_form']
        self.combine_sensors = args.build_args['combine_sensors']
        self.motion_augmentations = args.build_args['motion_augmentations']
        self.f_interp = args.build_args['f_interpolation']
        self.log_power = args.build_args['log_power']
        self.spectro_window = args.build_args['spectro_window']
        self.spectro_overlap = args.build_args['spectro_overlap']
        self.spectro_augmentations = args.build_args['spectro_augmentations']
        self.separated_channels = args.build_args['separated_channels']

        self.location_augment = args.build_args['location_augmentation']
        self.time_features = args.build_args['time_features']
        self.window_features = args.build_args['window_features']

        self.test_bag_position = args.build_args['test_bag_position']
        self.test_oversampling = args.build_args['test_oversampling']
        self.train_bag_position = args.build_args['train_bag_position']
        self.train_oversampling = args.build_args['train_oversampling']
        self.val_bag_position = args.build_args['val_bag_position']
        self.val_oversampling = args.build_args['val_oversampling']
        self.batch_size = args.build_args['batch_size']
        self.random = args.build_args['random']

        self.L = args.train_args['L']
        self.D = args.train_args['D']
        self.epochs = args.train_args['epochs']
        self.motion_epochs = args.train_args['motion_epochs']
        self.location_epochs = args.train_args['location_epochs']
        self.learning_rate = args.train_args['learning_rate']
        self.motion_transfer = args.train_args['motion_transfer']
        self.location_transfer = args.train_args['location_transfer']
        self.motorized = args.train_args['motorized']
        self.exclude_modes = args.train_args['exclude_modes']
        
        self.seq_thres = args.postprocess_args['seq_threshold']
        
        return

def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)
