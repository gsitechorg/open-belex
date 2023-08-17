r"""
By Dylon Edwards
"""

from typing import Dict

GVML_RN_REG_VALS: Dict[str, str] = {
    "RN_REG_0": "RN_REG_G0_ROW",
    "RN_REG_1": "RN_REG_G1_ROW",
    "RN_REG_2": "RN_REG_G2_ROW",
    "RN_REG_3": "RN_REG_G3_ROW",
    "RN_REG_4": "RN_REG_G4_ROW",
    "RN_REG_5": "RN_REG_G5_ROW",
    "RN_REG_6": "RN_REG_G6_ROW",
    "RN_REG_7": "RN_REG_G7_ROW",

    "RN_REG_8": "RN_REG_T0_ROW",
    "RN_REG_9": "RN_REG_T1_ROW",
    "RN_REG_10": "RN_REG_T2_ROW",
    "RN_REG_11": "RN_REG_T3_ROW",
    "RN_REG_12": "RN_REG_T4_ROW",
    "RN_REG_13": "RN_REG_T5_ROW",
    "RN_REG_14": "RN_REG_T6_ROW",

    "RN_REG_15": "RN_REG_FLAGS_ROW",
}

GVML_SM_REG_VALS: Dict[str, str] = {
    "SM_REG_4": "SM_0XFFFF_VAL",
    "SM_REG_5": "SM_0X0001_VAL",
    "SM_REG_6": "SM_0X1111_VAL",
    "SM_REG_7": "SM_0X0101_VAL",
    "SM_REG_8": "SM_0X000F_VAL",
    "SM_REG_9": "SM_0X0F0F_VAL",
    "SM_REG_10": "SM_0X0707_VAL",
    "SM_REG_11": "SM_0X5555_VAL",
    "SM_REG_12": "SM_0X3333_VAL",
    "SM_REG_13": "SM_0X00FF_VAL",
    "SM_REG_14": "SM_0X001F_VAL",
    "SM_REG_15": "SM_0X003F_VAL",
}

GVML_NYMS_BY_SM_REG_VAL: Dict[str, str] = {
    "SM_REG_4": "SM_0XFFFF",
    "SM_REG_5": "SM_0X0001",
    "SM_REG_6": "SM_0X1111",
    "SM_REG_7": "SM_0X0101",
    "SM_REG_8": "SM_0X000F",
    "SM_REG_9": "SM_0X0F0F",
    "SM_REG_10": "SM_0X0707",
    "SM_REG_11": "SM_0X5555",
    "SM_REG_12": "SM_0X3333",
    "SM_REG_13": "SM_0X00FF",
    "SM_REG_14": "SM_0X001F",
    "SM_REG_15": "SM_0X003F",
}

GVML_VALUES_BY_SM_REG_VAL: Dict[str, int] = {
    "SM_REG_4": 0xFFFF,
    "SM_REG_5": 0x0001,
    "SM_REG_6": 0x1111,
    "SM_REG_7": 0x0101,
    "SM_REG_8": 0x000F,
    "SM_REG_9": 0x0F0F,
    "SM_REG_10": 0x0707,
    "SM_REG_11": 0x5555,
    "SM_REG_12": 0x3333,
    "SM_REG_13": 0x00FF,
    "SM_REG_14": 0x001F,
    "SM_REG_15": 0x003F,
}
