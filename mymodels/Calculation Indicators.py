import os
import nibabel as nib
import numpy as np
from Function import dc, hd95, asd, obj_asd, hd
from glob import glob
from PIL import Image
import torch

from 超声心动图.超声心动图.ISIC.segment_3channel import dice_coefficient


def calculation_indicators(pre_list, label_list):
    """计算评价指标"""
    assert len(pre_list) == len(label_list)

    sum_dc = 0
    sum_asd = 0
    sum_hd = 0
    for i in range(len(pre_list)):
        pre = Image.open(pre_list[i]).convert('L')
        label = Image.open(label_list[i])
        pre = np.array(pre, dtype='int')
        label = np.array(label, dtype='int')
        result_dc = dc(pre, label)
        result_asd = asd(pre, label)
        result_hd = hd95(pre, label)
        sum_dc += result_dc
        sum_asd += result_asd
        sum_hd += result_hd
        print("第{}张图片的指标为：".format(i + 1), result_dc, result_asd, result_hd)
    return sum_dc / len(pre_list), sum_asd / len(pre_list), sum_hd / len(pre_list)


def main():
    pre_list = glob(r"F:\ISIC\output_test_png_finally_l\*.png")
    label_list = glob(r"F:\ISIC\test_label\labels\*.png")
    Dice, ASD, HD95 = calculation_indicators(pre_list, label_list)
    print("Dice:", Dice, "ASD:", ASD, "HD95:", HD95)


if __name__ == "__main__":
    main()
