TMD_MIL_parameters = {
    'extract_args': {
        'signals': ['Acc']
    },
    'resample_args': {
        'old_motion_fs': 100,
        'new_motion_fs': 10,
        'old_location_T': 1,
        'new_location_T': 60,
        'max_distance': 2,
        'mot_sampling_method': 'decimate',
        'loc_sampling_method': 'nearest'
    },
    'modify_args': {
        'loc_filter': None,
        'loc_filter_window': None,
        'loc_rescaler': None,
        'loc_virtual_features': ['velocity', 'acceleration'],
        'mot_smoother': None,
        'mot_smoother_window': 0,
        'mot_filter': None,
        'mot_filter_window': 0,
        'mot_rescaler': None,
        'mot_virtual_signals': ['acc_normXYZ', 'acc_jerk']
    },
    'segment_args': {
        'loc_length': 12,
        'loc_stride': 1,

        'mot_length': 600,
        'mot_stride': 600,

        'mot_bag_size': 3,
        'mot_bag_step': 600
    },
    'build_args': {
        'test_bag_position': 'same',
        'test_oversampling': True,
        'train_bag_position': 'same',
        'train_oversampling': False,
        'val_bag_position': 'same',
        'val_oversampling': True,
        'motion_features': ['acc_normXYZ', 'acc_jerk'],
        'in_bags': True,
        'get_position': True,
        'motion_form': 'spectrogram',
        'combine_sensors': 'concat',
        'motion_augmentations': None,
        'f_interpolation': 'log',
        'log_power': True,
        'spectro_window': 10,
        'spectro_overlap': 9,
        'spectro_augmentations': ['freq_mask', 'time_mask'],
        'location_augmentation': True,
        'time_features': ['velocity', 'acceleration'],
        'window_features': ['Movability', 'Mean', 'Std'],
        'batch_size': 32,
        'random': True
    },
    'train_args': {
        'L': 256,
        'D': 256,
        'epochs': 80,
        'motion_epochs': 80,
        'location_epochs': 200,
        'learning_rate': 0.0001,
        'motion_transfer': 'train',
        'location_transfer': 'skip',
        'motorized': False,
        'exclude_modes': ['undefined']
    }
}

Liang_parameters = {
    'extract_args': {
        'signals': ['Acc']
    },
    'resample_args': {
        'old_motion_fs': 100,
        'new_motion_fs': 50,
        'mot_sampling_method': 'step'
    },
    'modify_args': {
        'mot_smoother': 'moving_average',
        'mot_smoother_window': 5,
        'mot_filter': 'lowpass',
        'mot_filter_window': 0,
        'mot_rescaler': 'standard',
        'mot_virtual_signals': ['acc_normXYZ']
    },
    'segment_args': {
        'mot_length': 512,
        'mot_stride': 256,

        'mot_bag_size': 1,
        'mot_bag_step': 0
    },
    'build_args': {
        'test_bag_position': 'same',
        'test_oversampling': True,
        'train_bag_position': 'same',
        'train_oversampling': True,
        'val_bag_position': 'same',
        'val_oversampling': True,
        'motion_features': ['acc_normXYZ'],
        'in_bags': False,
        'get_position': False,
        'motion_form': 'temporal',
        'combine_sensors': 'concat',
        'motion_augmentations': None,
        'batch_size': 100,
        'random': True
    },
    'train_args': {
        'epochs': 80,
        'learning_rate': 0.0001,
        'motorized': False,
        'exclude_modes': ['undefined', 'run']
    }
}

Tang_parameters = {
    'extract_args': {
        'signals': ['Acc', 'Gyr', 'Mag']
    },
    'resample_args': {
        'old_motion_fs': 100,
        'new_motion_fs': 20,
        'mot_sampling_method': 'decimate'
    },
    'modify_args': {
        'mot_smoother': 'moving_average',
        'mot_smoother_window': 5,
        'mot_filter': None,
        'mot_filter_window': 0,
        'mot_rescaler': None,
        'mot_virtual_signals': ['acc_normXYZ', 'acc_jerkX', 'acc_jerkY', 'acc_jerkZ',
                                'mag_jerkX', 'mag_jerkY', 'mag_jerkZ',
                                'gyr_normXYZ']
    },
    'segment_args': {
        'mot_length': 1200,
        'mot_stride': 1200,

        'mot_bag_size': 1,
        'mot_bag_step': 0
    },
    'build_args': {
        'test_bag_position': 'same',
        'test_oversampling': True,
        'train_bag_position': 'same',
        'train_oversampling': True,
        'val_bag_position': 'same',
        'val_oversampling': True,
        'motion_features': ['acc_normXYZ', 'acc_jerkX', 'acc_jerkY', 'acc_jerkZ',
                            'mag_jerkX', 'mag_jerkY', 'mag_jerkZ',
                            'Gyr_x', 'Gyr_y', 'Gyr_z', 'gyr_normXYZ'],
        'in_bags': False,
        'get_position': False,
        'motion_form': 'temporal',
        'combine_sensors': 'separate',
        'separated_channels': [['acc_jerkX', 'acc_jerkY', 'acc_jerkZ'],
                               ['acc_normXYZ'],
                               ['mag_jerkX', 'mag_jerkY', 'mag_jerkZ'],
                               ['Gyr_x', 'Gyr_y', 'Gyr_z'],
                               ['gyr_normXYZ']],
        'motion_augmentations': None,
        'batch_size': 50
    },
    'train_args': {
        'epochs': 80,
        'learning_rate': 0.0001,
        'motorized': False,
        'exclude_modes': ['undefined']
    }
}

Wang_parameters = {
    'extract_args': {
        'signals': ['Lacc', 'Mag', 'Gyr', 'Bar']
    },
    'resample_args': {
        'old_motion_fs': 100,
        'new_motion_fs': 50,
        'mot_sampling_method': 'step'
    },
    'modify_args': {
        'mot_smoother': None,
        'mot_smoother_window': 5,
        'mot_filter': None,
        'mot_filter_window': 0,
        'mot_rescaler': 'standard',
        'mot_virtual_signals': []
    },
    'segment_args': {
        'mot_length': 450,
        'mot_stride': 450,

        'mot_bag_size': 1,
        'mot_bag_step': 0
    },
    'build_args': {
        'test_bag_position': 'same',
        'test_oversampling': True,
        'train_bag_position': 'same',
        'train_oversampling': True,
        'val_bag_position': 'same',
        'val_oversampling': True,
        'motion_features': ['Lacc_x', 'Lacc_y', 'Lacc_z',
                            'Gyr_x', 'Gyr_y', 'Gyr_z',
                            'Mag_x', 'Mag_y', 'Mag_z',
                            'Bar'],
        'in_bags': False,
        'get_position': False,
        'motion_form': 'temporal',
        'combine_sensors': 'separate',
        'separated_channels': [['Lacc_x', 'Lacc_y', 'Lacc_z'],
                               ['Gyr_x', 'Gyr_y', 'Gyr_z'],
                               ['Mag_x', 'Mag_y', 'Mag_z'],
                               ['Bar']],
        'motion_augmentations': None,
        'batch_size': 100
    },
    'train_args': {
        'epochs': 100,
        'learning_rate': 0.001,
        'motorized': False,
        'exclude_modes': ['undefined']
    }
}

SHL_preview_params = {
    'extract_args': {
        'dataset': 'SHL-preview'
    },
    'build_args': {
        'GPS_position': 'Hand',
        'train_test_split': 'loso',
        'train_test_hold_out': 1,
        'train_val_split': 'lopo_stratified',
        'train_val_hold_out': 0.15,
        'test_bag_position': 'same',
        'test_oversampling': True,
        'train_bag_position': 'same',
        'train_oversampling': True,
        'val_bag_position': 'same',
        'val_oversampling': True,
    }
}

SHL_complete_params = {
    'extract_args': {
        'dataset': 'SHL-complete'
    },
    'build_args': {
        'GPS_position': 'Hips',
        'train_test_split': 'ldo_random',
        'train_test_hold_out': 20,
        'train_val_split': 'ldo_random',
        'train_val_hold_out': 5,
        'test_bag_position': 'Hips',
        'test_oversampling': True,
        'train_bag_position': 'Hips',
        'train_oversampling': True,
        'val_bag_position': 'Hips',
        'val_oversampling': True,
    }
}
