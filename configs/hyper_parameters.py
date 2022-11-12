""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/2 10:15
"""

from collections import namedtuple

StateSpaceParameters = namedtuple('StateSpaceParameters', ['state_shape',
                                                         'obs_shape',
                                                         'entity_obs_shape',
                                                         'scalar_stat_dim',
                                                         'eneity_num',
                                                         'baseline_shape',
                                                         'n_cumulatscore'



                                                        ])

State_Space_Parameters = StateSpaceParameters(state_shape=[20, 163],
                                            obs_shape=[5, 67],
                                            baseline_shape=[36, ],
                                            entity_obs_shape=[100, ],
                                            scalar_stat_dim=150,
                                            eneity_num=20,
                                            n_cumulatscore=6
                                            )


ModelParameters = namedtuple('ModelParameters', ['hidden_size',
                                                 'layer_N',
                                                 'batch_size',
                                                 'sequence_length',
                                                 'core_hidden_dim',
                                                 'original_256',
                                                 'original_128',
                                                 'original_64',
                                                 'original_32',
                                                 'original_16',
                                                 'scalar_encoder_fc1_input',
                                                 'scalar_encoder_fc2_input',
                                                 'n_resblocks',
                                                 'context_size',
                                                 'autoregressive_embedding_size',
                                                 'entity_embedding_size',
                                                 'is_cnn',
                                                 'embedding_size',
                                                 'lstm_layers',
                                                 'baseline_input',
                                                 'lstm_input',
                                                 'scalar_context_dim'
                                                ])

Model_Parameters = ModelParameters(hidden_size=256,
                                   layer_N=2,
                                   batch_size=64,
                                   sequence_length=1,
                                   core_hidden_dim=256,
                                   original_256=256,
                                   original_128=128,
                                   original_64=64,
                                   original_32=32,
                                   original_16=16,
                                   scalar_encoder_fc1_input=192,
                                   scalar_encoder_fc2_input=192,
                                   scalar_context_dim=128,
                                   n_resblocks=4,
                                   context_size=128,
                                   autoregressive_embedding_size=256,
                                   entity_embedding_size=256,
                                   embedding_size=163,
                                   is_cnn=True,
                                   lstm_layers=1,
                                   baseline_input=352,
                                   lstm_input=384,




                                   )


ActionSpaceParameters = namedtuple('ActionSpaceParameters', ['level1_action_dim',
                                                             'location_dim',
                                                             'target_dim',
                                                             'attack_range',
                                                             'temperature',
                                                             'use_level1_action_type_mask',
                                                             'use_level2_action_type_mask',
                                                             ])

Action_Space_Parameters = ActionSpaceParameters(level1_action_dim=6,
                                                location_dim=15,
                                                target_dim=10,
                                                use_level1_action_type_mask=1,
                                                use_level2_action_type_mask=1,
                                                temperature=0.8,
                                                attack_range=60e3)
