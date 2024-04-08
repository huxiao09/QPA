# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Subsets of the CMU mocap database."""

from dm_control.locomotion.tasks.reference_pose import types

ClipCollection = types.ClipCollection

# get up
GET_UP = ClipCollection(
    ids=('CMU_139_16',
         'CMU_139_17',
         'CMU_139_18',
         'CMU_140_01',
         'CMU_140_02',
         'CMU_140_08',
         'CMU_140_09')
)

# Subset of about 40 minutes of varied locomotion behaviors.
LOCOMOTION_SMALL = ClipCollection(
    ids=('CMU_001_01',
         'CMU_002_03',
         'CMU_002_04',
         'CMU_009_01',
         'CMU_009_02',
         'CMU_009_03',
         'CMU_009_04',
         'CMU_009_05',
         'CMU_009_06',
         'CMU_009_07',
         'CMU_009_08',
         'CMU_009_09',
         'CMU_009_10',
         'CMU_009_11',
         'CMU_013_11',
         'CMU_013_13',
         'CMU_013_19',
         'CMU_013_32',
         'CMU_013_39',
         'CMU_013_40',
         'CMU_013_41',
         'CMU_013_42',
         'CMU_014_07',
         'CMU_014_08',
         'CMU_014_09',
         'CMU_016_01',
         'CMU_016_02',
         'CMU_016_03',
         'CMU_016_04',
         'CMU_016_05',
         'CMU_016_06',
         'CMU_016_07',
         'CMU_016_08',
         'CMU_016_09',
         'CMU_016_10',
         'CMU_016_17',
         'CMU_016_18',
         'CMU_016_19',
         'CMU_016_20',
         'CMU_016_27',
         'CMU_016_28',
         'CMU_016_29',
         'CMU_016_30',
         'CMU_016_35',
         'CMU_016_36',
         'CMU_016_37',
         'CMU_016_38',
         'CMU_016_39',
         'CMU_016_40',
         'CMU_016_41',
         'CMU_016_42',
         'CMU_016_43',
         'CMU_016_44',
         'CMU_016_45',
         'CMU_016_46',
         'CMU_016_48',
         'CMU_016_49',
         'CMU_016_50',
         'CMU_016_51',
         'CMU_016_52',
         'CMU_016_53',
         'CMU_016_54',
         'CMU_016_55',
         'CMU_016_56',
         'CMU_016_57',
         'CMU_035_17',
         'CMU_035_18',
         'CMU_035_19',
         'CMU_035_20',
         'CMU_035_21',
         'CMU_035_22',
         'CMU_035_23',
         'CMU_035_24',
         'CMU_035_25',
         'CMU_035_26',
         'CMU_036_02',
         'CMU_036_03',
         'CMU_036_09',
         'CMU_038_03',
         'CMU_038_04',
         'CMU_039_11',
         'CMU_047_01',
         'CMU_049_02',
         'CMU_049_03',
         'CMU_049_04',
         'CMU_049_05',
         'CMU_069_06',
         'CMU_069_07',
         'CMU_069_08',
         'CMU_069_09',
         'CMU_069_10',
         'CMU_069_11',
         'CMU_069_12',
         'CMU_069_13',
         'CMU_069_14',
         'CMU_069_15',
         'CMU_069_16',
         'CMU_069_17',
         'CMU_069_18',
         'CMU_069_19',
         'CMU_069_20',
         'CMU_069_21',
         'CMU_069_22',
         'CMU_069_23',
         'CMU_069_24',
         'CMU_069_25',
         'CMU_069_26',
         'CMU_069_27',
         'CMU_069_28',
         'CMU_069_29',
         'CMU_069_30',
         'CMU_069_31',
         'CMU_069_32',
         'CMU_069_33',
         'CMU_069_42',
         'CMU_069_43',
         'CMU_069_44',
         'CMU_069_45',
         'CMU_069_46',
         'CMU_069_47',
         'CMU_069_48',
         'CMU_069_49',
         'CMU_069_56',
         'CMU_069_57',
         'CMU_069_58',
         'CMU_069_59',
         'CMU_069_60',
         'CMU_069_61',
         'CMU_069_62',
         'CMU_069_63',
         'CMU_069_64',
         'CMU_069_65',
         'CMU_069_66',
         'CMU_069_67',
         'CMU_075_01',
         'CMU_075_02',
         'CMU_075_03',
         'CMU_075_04',
         'CMU_075_05',
         'CMU_075_06',
         'CMU_075_07',
         'CMU_075_08',
         'CMU_075_09',
         'CMU_075_10',
         'CMU_075_11',
         'CMU_075_12',
         'CMU_075_13',
         'CMU_075_14',
         'CMU_075_15',
         'CMU_076_10',
         'CMU_077_10',
         'CMU_077_11',
         'CMU_077_12',
         'CMU_077_13',
         'CMU_078_01',
         'CMU_078_02',
         'CMU_078_03',
         'CMU_078_07',
         'CMU_078_09',
         'CMU_078_10',
         'CMU_082_15',
         'CMU_083_36',
         'CMU_083_37',
         'CMU_083_38',
         'CMU_083_39',
         'CMU_083_40',
         'CMU_083_41',
         'CMU_083_42',
         'CMU_083_43',
         'CMU_083_45',
         'CMU_083_46',
         'CMU_083_48',
         'CMU_083_49',
         'CMU_083_51',
         'CMU_083_52',
         'CMU_083_53',
         'CMU_083_54',
         'CMU_083_56',
         'CMU_083_57',
         'CMU_083_58',
         'CMU_083_59',
         'CMU_083_60',
         'CMU_083_61',
         'CMU_083_62',
         'CMU_083_64',
         'CMU_083_65',
         'CMU_086_01',
         'CMU_086_11',
         'CMU_090_06',
         'CMU_090_07',
         'CMU_091_39',
         'CMU_091_40',
         'CMU_091_41',
         'CMU_091_42',
         'CMU_091_43',
         'CMU_091_44',
         'CMU_091_45',
         'CMU_091_46',
         'CMU_091_47',
         'CMU_091_48',
         'CMU_091_49',
         'CMU_091_50',
         'CMU_091_51',
         'CMU_091_52',
         'CMU_091_53',
         'CMU_104_53',
         'CMU_104_54',
         'CMU_104_55',
         'CMU_104_56',
         'CMU_104_57',
         'CMU_105_39',
         'CMU_105_40',
         'CMU_105_41',
         'CMU_105_42',
         'CMU_105_43',
         'CMU_105_44',
         'CMU_105_45',
         'CMU_105_46',
         'CMU_105_47',
         'CMU_105_48',
         'CMU_105_49',
         'CMU_105_50',
         'CMU_105_51',
         'CMU_105_52',
         'CMU_118_01',
         'CMU_118_02',
         'CMU_118_03',
         'CMU_118_04',
         'CMU_118_05',
         'CMU_118_06',
         'CMU_118_07',
         'CMU_118_08',
         'CMU_118_09',
         'CMU_118_10',
         'CMU_118_11',
         'CMU_118_12',
         'CMU_118_13',
         'CMU_118_14',
         'CMU_118_15',
         'CMU_118_16',
         'CMU_118_17',
         'CMU_118_18',
         'CMU_118_19',
         'CMU_118_20',
         'CMU_118_21',
         'CMU_118_22',
         'CMU_118_23',
         'CMU_118_24',
         'CMU_118_25',
         'CMU_118_26',
         'CMU_118_27',
         'CMU_118_28',
         'CMU_118_29',
         'CMU_118_30',
         'CMU_127_03',
         'CMU_127_04',
         'CMU_127_05',
         'CMU_127_06',
         'CMU_127_07',
         'CMU_127_08',
         'CMU_127_09',
         'CMU_127_10',
         'CMU_127_11',
         'CMU_127_12',
         'CMU_127_13',
         'CMU_127_14',
         'CMU_127_15',
         'CMU_127_16',
         'CMU_127_17',
         'CMU_127_18',
         'CMU_127_19',
         'CMU_127_20',
         'CMU_127_21',
         'CMU_127_22',
         'CMU_127_23',
         'CMU_127_24',
         'CMU_127_25',
         'CMU_127_26',
         'CMU_127_27',
         'CMU_127_28',
         'CMU_127_29',
         'CMU_127_30',
         'CMU_127_31',
         'CMU_127_32',
         'CMU_127_37',
         'CMU_127_38',
         'CMU_128_02',
         'CMU_128_03',
         'CMU_128_04',
         'CMU_128_05',
         'CMU_128_06',
         'CMU_128_07',
         'CMU_128_08',
         'CMU_128_09',
         'CMU_128_10',
         'CMU_128_11',
         'CMU_132_23',
         'CMU_132_24',
         'CMU_132_25',
         'CMU_132_26',
         'CMU_132_27',
         'CMU_132_28',
         'CMU_139_10',
         'CMU_139_11',
         'CMU_139_12',
         'CMU_139_13',
         'CMU_143_01',
         'CMU_143_02',
         'CMU_143_03',
         'CMU_143_04',
         'CMU_143_05',
         'CMU_143_06',
         'CMU_143_07',
         'CMU_143_08',
         'CMU_143_09',
         'CMU_143_42'))

# Subset of about 2 minutes of walking behaviors.
WALK_TINY = ClipCollection(
    ids=('CMU_016_22',
         'CMU_016_23',
         'CMU_016_24',
         'CMU_016_25',
         'CMU_016_26',
         'CMU_016_27',
         'CMU_016_28',
         'CMU_016_29',
         'CMU_016_30',
         'CMU_016_31',
         'CMU_016_32',
         'CMU_016_33',
         'CMU_016_34',
         'CMU_016_47',
         'CMU_016_58',
         'CMU_047_01',
         'CMU_056_01',
         'CMU_069_01',
         'CMU_069_02',
         'CMU_069_03',
         'CMU_069_04',
         'CMU_069_05',
         'CMU_069_20',
         'CMU_069_21',
         'CMU_069_22',
         'CMU_069_23',
         'CMU_069_24',
         'CMU_069_25',
         'CMU_069_26',
         'CMU_069_27',
         'CMU_069_28',
         'CMU_069_29',
         'CMU_069_30',
         'CMU_069_31',
         'CMU_069_32',
         'CMU_069_33'))

# Subset of about 2 minutes of walking/running/jumping behaviors.
RUN_JUMP_TINY = ClipCollection(
    ids=('CMU_009_01',
         'CMU_009_02',
         'CMU_009_03',
         'CMU_009_04',
         'CMU_009_05',
         'CMU_009_06',
         'CMU_009_07',
         'CMU_009_08',
         'CMU_009_09',
         'CMU_009_10',
         'CMU_009_11',
         'CMU_016_22',
         'CMU_016_23',
         'CMU_016_24',
         'CMU_016_25',
         'CMU_016_26',
         'CMU_016_27',
         'CMU_016_28',
         'CMU_016_29',
         'CMU_016_30',
         'CMU_016_31',
         'CMU_016_32',
         'CMU_016_47',
         'CMU_016_48',
         'CMU_016_49',
         'CMU_016_50',
         'CMU_016_55',
         'CMU_016_58',
         'CMU_049_04',
         'CMU_049_05',
         'CMU_069_01',
         'CMU_069_02',
         'CMU_069_03',
         'CMU_069_04',
         'CMU_069_05',
         'CMU_075_01',
         'CMU_075_02',
         'CMU_075_03',
         'CMU_075_10',
         'CMU_075_11',
         'CMU_127_03',
         'CMU_127_06',
         'CMU_127_07',
         'CMU_127_08',
         'CMU_127_09',
         'CMU_127_10',
         'CMU_127_11',
         'CMU_127_12',
         'CMU_128_02',
         'CMU_128_03'))

# Subset of about 3.5 hours of varied locomotion behaviors and hand movements.
ALL = ClipCollection(
    ids=('CMU_001_01',
         'CMU_002_01',
         'CMU_002_02',
         'CMU_002_03',
         'CMU_002_04',
         'CMU_005_01',
         'CMU_006_01',
         'CMU_006_02',
         'CMU_006_03',
         'CMU_006_04',
         'CMU_006_05',
         'CMU_006_06',
         'CMU_006_07',
         'CMU_006_08',
         'CMU_006_09',
         'CMU_006_10',
         'CMU_006_11',
         'CMU_006_12',
         'CMU_006_13',
         'CMU_006_14',
         'CMU_006_15',
         'CMU_007_01',
         'CMU_007_02',
         'CMU_007_03',
         'CMU_007_04',
         'CMU_007_05',
         'CMU_007_06',
         'CMU_007_07',
         'CMU_007_08',
         'CMU_007_09',
         'CMU_007_10',
         'CMU_007_11',
         'CMU_007_12',
         'CMU_008_01',
         'CMU_008_02',
         'CMU_008_03',
         'CMU_008_04',
         'CMU_008_05',
         'CMU_008_06',
         'CMU_008_07',
         'CMU_008_08',
         'CMU_008_09',
         'CMU_008_10',
         'CMU_008_11',
         'CMU_009_01',
         'CMU_009_02',
         'CMU_009_03',
         'CMU_009_04',
         'CMU_009_05',
         'CMU_009_06',
         'CMU_009_07',
         'CMU_009_08',
         'CMU_009_09',
         'CMU_009_10',
         'CMU_009_11',
         'CMU_009_12',
         'CMU_010_04',
         'CMU_013_11',
         'CMU_013_13',
         'CMU_013_19',
         'CMU_013_26',
         'CMU_013_27',
         'CMU_013_28',
         'CMU_013_29',
         'CMU_013_30',
         'CMU_013_31',
         'CMU_013_32',
         'CMU_013_39',
         'CMU_013_40',
         'CMU_013_41',
         'CMU_013_42',
         'CMU_014_06',
         'CMU_014_07',
         'CMU_014_08',
         'CMU_014_09',
         'CMU_014_14',
         'CMU_014_20',
         'CMU_014_24',
         'CMU_014_25',
         'CMU_014_26',
         'CMU_015_01',
         'CMU_015_03',
         'CMU_015_04',
         'CMU_015_05',
         'CMU_015_06',
         'CMU_015_07',
         'CMU_015_08',
         'CMU_015_09',
         'CMU_015_12',
         'CMU_015_14',
         'CMU_016_01',
         'CMU_016_02',
         'CMU_016_03',
         'CMU_016_04',
         'CMU_016_05',
         'CMU_016_06',
         'CMU_016_07',
         'CMU_016_08',
         'CMU_016_09',
         'CMU_016_10',
         'CMU_016_11',
         'CMU_016_12',
         'CMU_016_13',
         'CMU_016_14',
         'CMU_016_15',
         'CMU_016_16',
         'CMU_016_17',
         'CMU_016_18',
         'CMU_016_19',
         'CMU_016_20',
         'CMU_016_21',
         'CMU_016_22',
         'CMU_016_23',
         'CMU_016_24',
         'CMU_016_25',
         'CMU_016_26',
         'CMU_016_27',
         'CMU_016_28',
         'CMU_016_29',
         'CMU_016_30',
         'CMU_016_31',
         'CMU_016_32',
         'CMU_016_33',
         'CMU_016_34',
         'CMU_016_35',
         'CMU_016_36',
         'CMU_016_37',
         'CMU_016_38',
         'CMU_016_39',
         'CMU_016_40',
         'CMU_016_41',
         'CMU_016_42',
         'CMU_016_43',
         'CMU_016_44',
         'CMU_016_45',
         'CMU_016_46',
         'CMU_016_47',
         'CMU_016_48',
         'CMU_016_49',
         'CMU_016_50',
         'CMU_016_51',
         'CMU_016_52',
         'CMU_016_53',
         'CMU_016_54',
         'CMU_016_55',
         'CMU_016_56',
         'CMU_016_57',
         'CMU_016_58',
         'CMU_017_01',
         'CMU_017_02',
         'CMU_017_03',
         'CMU_017_04',
         'CMU_017_05',
         'CMU_017_06',
         'CMU_017_07',
         'CMU_017_08',
         'CMU_017_09',
         'CMU_017_10',
         'CMU_024_01',
         'CMU_025_01',
         'CMU_026_01',
         'CMU_026_02',
         'CMU_026_03',
         'CMU_026_04',
         'CMU_026_05',
         'CMU_026_06',
         'CMU_026_07',
         'CMU_026_08',
         'CMU_027_01',
         'CMU_027_02',
         'CMU_027_03',
         'CMU_027_04',
         'CMU_027_05',
         'CMU_027_06',
         'CMU_027_07',
         'CMU_027_08',
         'CMU_027_09',
         'CMU_029_01',
         'CMU_029_02',
         'CMU_029_03',
         'CMU_029_04',
         'CMU_029_05',
         'CMU_029_06',
         'CMU_029_07',
         'CMU_029_08',
         'CMU_029_09',
         'CMU_029_10',
         'CMU_029_11',
         'CMU_029_12',
         'CMU_029_13',
         'CMU_031_01',
         'CMU_031_02',
         'CMU_031_03',
         'CMU_031_06',
         'CMU_031_07',
         'CMU_031_08',
         'CMU_032_01',
         'CMU_032_02',
         'CMU_032_03',
         'CMU_032_04',
         'CMU_032_05',
         'CMU_032_06',
         'CMU_032_07',
         'CMU_032_08',
         'CMU_032_09',
         'CMU_032_10',
         'CMU_032_11',
         'CMU_035_01',
         'CMU_035_02',
         'CMU_035_03',
         'CMU_035_04',
         'CMU_035_05',
         'CMU_035_06',
         'CMU_035_07',
         'CMU_035_08',
         'CMU_035_09',
         'CMU_035_10',
         'CMU_035_11',
         'CMU_035_12',
         'CMU_035_13',
         'CMU_035_14',
         'CMU_035_15',
         'CMU_035_16',
         'CMU_035_17',
         'CMU_035_18',
         'CMU_035_19',
         'CMU_035_20',
         'CMU_035_21',
         'CMU_035_22',
         'CMU_035_23',
         'CMU_035_24',
         'CMU_035_25',
         'CMU_035_26',
         'CMU_035_27',
         'CMU_035_28',
         'CMU_035_29',
         'CMU_035_30',
         'CMU_035_31',
         'CMU_035_32',
         'CMU_035_33',
         'CMU_035_34',
         'CMU_036_02',
         'CMU_036_03',
         'CMU_036_09',
         'CMU_037_01',
         'CMU_038_01',
         'CMU_038_02',
         'CMU_038_03',
         'CMU_038_04',
         'CMU_039_11',
         'CMU_040_02',
         'CMU_040_03',
         'CMU_040_04',
         'CMU_040_05',
         'CMU_040_10',
         'CMU_040_11',
         'CMU_040_12',
         'CMU_041_02',
         'CMU_041_03',
         'CMU_041_04',
         'CMU_041_05',
         'CMU_041_06',
         'CMU_041_10',
         'CMU_041_11',
         'CMU_045_01',
         'CMU_046_01',
         'CMU_047_01',
         'CMU_049_01',
         'CMU_049_02',
         'CMU_049_03',
         'CMU_049_04',
         'CMU_049_05',
         'CMU_049_06',
         'CMU_049_07',
         'CMU_049_08',
         'CMU_049_09',
         'CMU_049_10',
         'CMU_049_11',
         'CMU_049_12',
         'CMU_049_13',
         'CMU_049_14',
         'CMU_049_15',
         'CMU_049_16',
         'CMU_049_17',
         'CMU_049_18',
         'CMU_049_19',
         'CMU_049_20',
         'CMU_049_22',
         'CMU_056_01',
         'CMU_056_04',
         'CMU_056_05',
         'CMU_056_06',
         'CMU_056_07',
         'CMU_056_08',
         'CMU_060_02',
         'CMU_060_03',
         'CMU_060_05',
         'CMU_060_12',
         'CMU_060_14',
         'CMU_061_01',
         'CMU_061_02',
         'CMU_061_03',
         'CMU_061_04',
         'CMU_061_05',
         'CMU_061_06',
         'CMU_061_07',
         'CMU_061_08',
         'CMU_061_09',
         'CMU_061_10',
         'CMU_061_15',
         'CMU_069_01',
         'CMU_069_02',
         'CMU_069_03',
         'CMU_069_04',
         'CMU_069_05',
         'CMU_069_06',
         'CMU_069_07',
         'CMU_069_08',
         'CMU_069_09',
         'CMU_069_10',
         'CMU_069_11',
         'CMU_069_12',
         'CMU_069_13',
         'CMU_069_14',
         'CMU_069_15',
         'CMU_069_16',
         'CMU_069_17',
         'CMU_069_18',
         'CMU_069_19',
         'CMU_069_20',
         'CMU_069_21',
         'CMU_069_22',
         'CMU_069_23',
         'CMU_069_24',
         'CMU_069_25',
         'CMU_069_26',
         'CMU_069_27',
         'CMU_069_28',
         'CMU_069_29',
         'CMU_069_30',
         'CMU_069_31',
         'CMU_069_32',
         'CMU_069_33',
         'CMU_069_34',
         'CMU_069_36',
         'CMU_069_37',
         'CMU_069_38',
         'CMU_069_39',
         'CMU_069_40',
         'CMU_069_41',
         'CMU_069_42',
         'CMU_069_43',
         'CMU_069_44',
         'CMU_069_45',
         'CMU_069_46',
         'CMU_069_47',
         'CMU_069_48',
         'CMU_069_49',
         'CMU_069_50',
         'CMU_069_51',
         'CMU_069_52',
         'CMU_069_53',
         'CMU_069_54',
         'CMU_069_55',
         'CMU_069_56',
         'CMU_069_57',
         'CMU_069_58',
         'CMU_069_59',
         'CMU_069_60',
         'CMU_069_61',
         'CMU_069_62',
         'CMU_069_63',
         'CMU_069_64',
         'CMU_069_65',
         'CMU_069_66',
         'CMU_069_67',
         'CMU_075_01',
         'CMU_075_02',
         'CMU_075_03',
         'CMU_075_04',
         'CMU_075_05',
         'CMU_075_06',
         'CMU_075_07',
         'CMU_075_08',
         'CMU_075_09',
         'CMU_075_10',
         'CMU_075_11',
         'CMU_075_12',
         'CMU_075_13',
         'CMU_075_14',
         'CMU_075_15',
         'CMU_076_01',
         'CMU_076_02',
         'CMU_076_06',
         'CMU_076_08',
         'CMU_076_09',
         'CMU_076_10',
         'CMU_076_11',
         'CMU_077_02',
         'CMU_077_03',
         'CMU_077_04',
         'CMU_077_10',
         'CMU_077_11',
         'CMU_077_12',
         'CMU_077_13',
         'CMU_077_14',
         'CMU_077_15',
         'CMU_077_16',
         'CMU_077_17',
         'CMU_077_18',
         'CMU_077_21',
         'CMU_077_27',
         'CMU_077_28',
         'CMU_077_29',
         'CMU_077_30',
         'CMU_077_31',
         'CMU_077_32',
         'CMU_077_33',
         'CMU_077_34',
         'CMU_078_01',
         'CMU_078_02',
         'CMU_078_03',
         'CMU_078_04',
         'CMU_078_05',
         'CMU_078_07',
         'CMU_078_09',
         'CMU_078_10',
         'CMU_078_13',
         'CMU_078_14',
         'CMU_078_15',
         'CMU_078_16',
         'CMU_078_17',
         'CMU_078_18',
         'CMU_078_19',
         'CMU_078_20',
         'CMU_078_21',
         'CMU_078_22',
         'CMU_078_23',
         'CMU_078_24',
         'CMU_078_25',
         'CMU_078_26',
         'CMU_078_27',
         'CMU_078_28',
         'CMU_078_29',
         'CMU_078_30',
         'CMU_078_31',
         'CMU_078_32',
         'CMU_078_33',
         'CMU_082_08',
         'CMU_082_09',
         'CMU_082_10',
         'CMU_082_11',
         'CMU_082_14',
         'CMU_082_15',
         'CMU_083_18',
         'CMU_083_19',
         'CMU_083_20',
         'CMU_083_21',
         'CMU_083_33',
         'CMU_083_36',
         'CMU_083_37',
         'CMU_083_38',
         'CMU_083_39',
         'CMU_083_40',
         'CMU_083_41',
         'CMU_083_42',
         'CMU_083_43',
         'CMU_083_44',
         'CMU_083_45',
         'CMU_083_46',
         'CMU_083_48',
         'CMU_083_49',
         'CMU_083_51',
         'CMU_083_52',
         'CMU_083_53',
         'CMU_083_54',
         'CMU_083_55',
         'CMU_083_56',
         'CMU_083_57',
         'CMU_083_58',
         'CMU_083_59',
         'CMU_083_60',
         'CMU_083_61',
         'CMU_083_62',
         'CMU_083_63',
         'CMU_083_64',
         'CMU_083_65',
         'CMU_083_66',
         'CMU_083_67',
         'CMU_086_01',
         'CMU_086_02',
         'CMU_086_03',
         'CMU_086_07',
         'CMU_086_08',
         'CMU_086_11',
         'CMU_086_14',
         'CMU_090_06',
         'CMU_090_07',
         'CMU_091_01',
         'CMU_091_02',
         'CMU_091_03',
         'CMU_091_04',
         'CMU_091_05',
         'CMU_091_06',
         'CMU_091_07',
         'CMU_091_08',
         'CMU_091_10',
         'CMU_091_11',
         'CMU_091_12',
         'CMU_091_13',
         'CMU_091_14',
         'CMU_091_15',
         'CMU_091_16',
         'CMU_091_17',
         'CMU_091_18',
         'CMU_091_19',
         'CMU_091_20',
         'CMU_091_21',
         'CMU_091_22',
         'CMU_091_23',
         'CMU_091_24',
         'CMU_091_25',
         'CMU_091_26',
         'CMU_091_27',
         'CMU_091_28',
         'CMU_091_29',
         'CMU_091_30',
         'CMU_091_31',
         'CMU_091_32',
         'CMU_091_33',
         'CMU_091_34',
         'CMU_091_35',
         'CMU_091_36',
         'CMU_091_37',
         'CMU_091_38',
         'CMU_091_39',
         'CMU_091_40',
         'CMU_091_41',
         'CMU_091_42',
         'CMU_091_43',
         'CMU_091_44',
         'CMU_091_45',
         'CMU_091_46',
         'CMU_091_47',
         'CMU_091_48',
         'CMU_091_49',
         'CMU_091_50',
         'CMU_091_51',
         'CMU_091_52',
         'CMU_091_53',
         'CMU_091_54',
         'CMU_091_55',
         'CMU_091_56',
         'CMU_091_57',
         'CMU_091_58',
         'CMU_091_59',
         'CMU_091_60',
         'CMU_091_61',
         'CMU_091_62',
         'CMU_104_53',
         'CMU_104_54',
         'CMU_104_55',
         'CMU_104_56',
         'CMU_104_57',
         'CMU_105_01',
         'CMU_105_02',
         'CMU_105_03',
         'CMU_105_04',
         'CMU_105_05',
         'CMU_105_07',
         'CMU_105_08',
         'CMU_105_10',
         'CMU_105_17',
         'CMU_105_18',
         'CMU_105_19',
         'CMU_105_20',
         'CMU_105_22',
         'CMU_105_29',
         'CMU_105_31',
         'CMU_105_34',
         'CMU_105_36',
         'CMU_105_37',
         'CMU_105_38',
         'CMU_105_39',
         'CMU_105_40',
         'CMU_105_41',
         'CMU_105_42',
         'CMU_105_43',
         'CMU_105_44',
         'CMU_105_45',
         'CMU_105_46',
         'CMU_105_47',
         'CMU_105_48',
         'CMU_105_49',
         'CMU_105_50',
         'CMU_105_51',
         'CMU_105_52',
         'CMU_105_53',
         'CMU_105_54',
         'CMU_105_55',
         'CMU_105_56',
         'CMU_105_57',
         'CMU_105_58',
         'CMU_105_59',
         'CMU_105_60',
         'CMU_105_61',
         'CMU_105_62',
         'CMU_107_01',
         'CMU_107_02',
         'CMU_107_03',
         'CMU_107_04',
         'CMU_107_05',
         'CMU_107_06',
         'CMU_107_07',
         'CMU_107_08',
         'CMU_107_09',
         'CMU_107_11',
         'CMU_107_12',
         'CMU_107_13',
         'CMU_107_14',
         'CMU_108_01',
         'CMU_108_02',
         'CMU_108_03',
         'CMU_108_04',
         'CMU_108_05',
         'CMU_108_06',
         'CMU_108_07',
         'CMU_108_08',
         'CMU_108_09',
         'CMU_108_12',
         'CMU_108_13',
         'CMU_108_14',
         'CMU_108_17',
         'CMU_108_18',
         'CMU_108_19',
         'CMU_108_20',
         'CMU_108_21',
         'CMU_108_22',
         'CMU_108_23',
         'CMU_108_24',
         'CMU_108_25',
         'CMU_108_26',
         'CMU_108_27',
         'CMU_108_28',
         'CMU_114_13',
         'CMU_114_14',
         'CMU_114_15',
         'CMU_118_01',
         'CMU_118_02',
         'CMU_118_03',
         'CMU_118_04',
         'CMU_118_05',
         'CMU_118_06',
         'CMU_118_07',
         'CMU_118_08',
         'CMU_118_09',
         'CMU_118_10',
         'CMU_118_11',
         'CMU_118_12',
         'CMU_118_13',
         'CMU_118_14',
         'CMU_118_15',
         'CMU_118_16',
         'CMU_118_17',
         'CMU_118_18',
         'CMU_118_19',
         'CMU_118_20',
         'CMU_118_21',
         'CMU_118_22',
         'CMU_118_23',
         'CMU_118_24',
         'CMU_118_25',
         'CMU_118_26',
         'CMU_118_27',
         'CMU_118_28',
         'CMU_118_29',
         'CMU_118_30',
         'CMU_118_32',
         'CMU_120_20',
         'CMU_124_03',
         'CMU_124_04',
         'CMU_124_05',
         'CMU_124_06',
         'CMU_127_02',
         'CMU_127_03',
         'CMU_127_04',
         'CMU_127_05',
         'CMU_127_06',
         'CMU_127_07',
         'CMU_127_08',
         'CMU_127_09',
         'CMU_127_10',
         'CMU_127_11',
         'CMU_127_12',
         'CMU_127_13',
         'CMU_127_14',
         'CMU_127_15',
         'CMU_127_16',
         'CMU_127_17',
         'CMU_127_18',
         'CMU_127_19',
         'CMU_127_20',
         'CMU_127_21',
         'CMU_127_22',
         'CMU_127_23',
         'CMU_127_24',
         'CMU_127_25',
         'CMU_127_26',
         'CMU_127_27',
         'CMU_127_28',
         'CMU_127_29',
         'CMU_127_30',
         'CMU_127_31',
         'CMU_127_32',
         'CMU_127_37',
         'CMU_127_38',
         'CMU_128_02',
         'CMU_128_03',
         'CMU_128_04',
         'CMU_128_05',
         'CMU_128_06',
         'CMU_128_07',
         'CMU_128_08',
         'CMU_128_09',
         'CMU_128_10',
         'CMU_128_11',
         'CMU_132_01',
         'CMU_132_02',
         'CMU_132_03',
         'CMU_132_04',
         'CMU_132_05',
         'CMU_132_06',
         'CMU_132_07',
         'CMU_132_08',
         'CMU_132_09',
         'CMU_132_10',
         'CMU_132_11',
         'CMU_132_12',
         'CMU_132_13',
         'CMU_132_14',
         'CMU_132_15',
         'CMU_132_16',
         'CMU_132_17',
         'CMU_132_18',
         'CMU_132_19',
         'CMU_132_20',
         'CMU_132_21',
         'CMU_132_22',
         'CMU_132_23',
         'CMU_132_24',
         'CMU_132_25',
         'CMU_132_26',
         'CMU_132_27',
         'CMU_132_28',
         'CMU_132_29',
         'CMU_132_30',
         'CMU_132_31',
         'CMU_132_32',
         'CMU_132_33',
         'CMU_132_34',
         'CMU_132_35',
         'CMU_132_36',
         'CMU_132_37',
         'CMU_132_38',
         'CMU_132_39',
         'CMU_132_40',
         'CMU_132_41',
         'CMU_132_42',
         'CMU_132_43',
         'CMU_132_44',
         'CMU_132_45',
         'CMU_132_46',
         'CMU_132_47',
         'CMU_132_48',
         'CMU_132_49',
         'CMU_132_50',
         'CMU_132_51',
         'CMU_132_52',
         'CMU_132_53',
         'CMU_132_54',
         'CMU_132_55',
         'CMU_133_03',
         'CMU_133_04',
         'CMU_133_05',
         'CMU_133_06',
         'CMU_133_07',
         'CMU_133_08',
         'CMU_133_10',
         'CMU_133_11',
         'CMU_133_12',
         'CMU_133_13',
         'CMU_133_14',
         'CMU_133_15',
         'CMU_133_16',
         'CMU_133_17',
         'CMU_133_18',
         'CMU_133_19',
         'CMU_133_20',
         'CMU_133_21',
         'CMU_133_22',
         'CMU_133_23',
         'CMU_133_24',
         'CMU_139_04',
         'CMU_139_10',
         'CMU_139_11',
         'CMU_139_12',
         'CMU_139_13',
         'CMU_139_14',
         'CMU_139_15',
         'CMU_139_16',
         'CMU_139_17',
         'CMU_139_18',
         'CMU_139_21',
         'CMU_139_28',
         'CMU_140_01',
         'CMU_140_02',
         'CMU_140_08',
         'CMU_140_09',
         'CMU_143_01',
         'CMU_143_02',
         'CMU_143_03',
         'CMU_143_04',
         'CMU_143_05',
         'CMU_143_06',
         'CMU_143_07',
         'CMU_143_08',
         'CMU_143_09',
         'CMU_143_14',
         'CMU_143_15',
         'CMU_143_16',
         'CMU_143_29',
         'CMU_143_32',
         'CMU_143_39',
         'CMU_143_40',
         'CMU_143_41',
         'CMU_143_42'))


CMU_SUBSETS_DICT = dict(
    walk_tiny=WALK_TINY,
    run_jump_tiny=RUN_JUMP_TINY,
    get_up=GET_UP,
    locomotion_small=LOCOMOTION_SMALL,
    all=ALL
    )