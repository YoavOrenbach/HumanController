
pose_dictionary = {
    "bow": 0,
    "circle": 1,
    "cross": 2,
    "crossarm": 3,
    "down-arrow": 4,
    "L-down": 5,
    "L-left": 6,
    "L-right": 7,
    "L-up": 8,
    "LB": 9,
    "left-arrow": 10,
    "LT": 11,
    "normal": 12,
    "options": 13,
    "praise": 14,
    "R-down": 15,
    "R-left": 16,
    "R-right": 17,
    "R-up": 18,
    "RB": 19,
    "right-arrow": 20,
    "RT": 21,
    "square": 22,
    "triangle": 23,
    "up-arrow": 24
}

test1_pose_switches = [7,8,9,20,21,22,38,39,55,56,57,68,69,75,76,77,87,99,100,105,106,107,122,131,132,138,139,159,160,
                       183,187,188,211,212,213,214,215,233,234,235,236,241,242,243,260,261,262,263,264,265,266,277,278,
                       279,280,281,294,295,296,297,298,304,305,306,316,317,318,319,320,321,325,326,327,344,345,346,347,
                       348,364,365,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389]


def test1_movements(img_num):
    class_num = 0
    if 0 <= img_num <= 8:
        class_num = 12
    elif 9 <= img_num <= 20:
        class_num = 22
    elif 21 <= img_num <= 38:
        class_num = 23
    elif 39 <= img_num <= 55:
        class_num = 2
    elif 56 <= img_num <= 68:
        class_num = 1
    elif 69 <= img_num <= 76:
        class_num = 12
    elif 77 <= img_num <= 86:
        class_num = 11
    elif 87 <= img_num <= 99:
        class_num = 9
    elif 100 <= img_num <= 106:
        class_num = 12
    elif 107 <= img_num <= 121:
        class_num = 21
    elif 122 <= img_num <= 130:
        class_num = 19
    elif 131 <= img_num <= 138:
        class_num = 12
    elif 139 <= img_num <= 159:
        class_num = 7
    elif 160 <= img_num <= 182:
        class_num = 8
    elif 183 <= img_num <= 186:
        class_num = 12
    elif 187 <= img_num <= 211:
        class_num = 6
    elif 212 <= img_num <= 214:
        class_num = 12
    elif 215 <= img_num <= 233:
        class_num = 5
    elif 234 <= img_num <= 242:
        class_num = 12
    elif 243 <= img_num <= 262:
        class_num = 18
    elif 263 <= img_num <= 279:
        class_num = 15
    elif 280 <= img_num <= 294:
        class_num = 16
    elif 295 <= img_num <= 297:
        class_num = 18
    elif 298 <= img_num <= 304:
        class_num = 12
    elif 305 <= img_num <= 318:
        class_num = 17
    elif 319 <= img_num <= 326:
        class_num = 12
    elif 327 <= img_num <= 344:
        class_num = 14
    elif 345 <= img_num <= 346:
        class_num = 24
    elif 347 <= img_num <= 364:
        class_num = 3
    elif 365 <= img_num <= 375:
        class_num = 12
    elif 375 <= img_num <= 407:
        class_num = 13
    return class_num


test2_pose_switches = list(range(61, 68)) + list(range(147, 153)) + list(range(251, 255)) + list(range(360, 377)) + \
                      list(range(510, 515)) + list(range(542, 550)) + list(range(611, 620)) + list(range(634, 640)) + \
                      list(range(710, 716)) + list(range(776, 781)) + list(range(789, 798)) + list(range(831, 836)) +\
                      list(range(883, 888)) + list(range(898, 909)) + list(range(961, 969)) + list(range(1011, 1014)) +\
                      list(range(1096,1103)) + list(range(1175,1184)) + list(range(1195,1202)) + list(range(1318,1325)) +\
                      list(range(1328,1336)) + list(range(1389,1395)) + list(range(1410,1421)) + list(range(1472,1482)) +\
                      list(range(1503,1508)) + list(range(1559,1564)) + list(range(1583,1593)) + list(range(1703,1727)) +\
                      list(range(1823,1861)) + list(range(1968,1989)) + list(range(2064,2076)) + list(range(2102,2107)) + \
                      list(range(2204,2214)) + list(range(2298,2307)) + list(range(2390,2403)) + list(range(2498,2508)) +\
                      list(range(2524,2533)) + list(range(2583,2591)) + list(range(2602,2609)) + list(range(2648,2658)) +\
                      list(range(2682,2690)) + list(range(2822,2832)) + list(range(2866,2883)) + list(range(3018,3036)) +\
                      list(range(3051,3061)) + list(range(3146,3156)) + list(range(3172,3181)) + list(range(3224,3234)) +\
                      list(range(3273,3307)) + list(range(3553,3586)) + list(range(3634,3645)) + list(range(3710,3719)) +\
                      list(range(3803,3813)) + list(range(3847,3858)) + list(range(3928,3938)) + list(range(3956,3961)) +\
                      list(range(4073,4077)) + list(range(4097,4102)) + list(range(4147,4156)) + list(range(4204,4209)) +\
                      list(range(4337,4343)) + list(range(4396,4403)) + list(range(4504,4514)) + list(range(4560,4567)) +\
                      list(range(4636,4643)) + list(range(4656,4666)) + list(range(4743,4752)) + list(range(4813,4819)) +\
                      list(range(4959,4967)) + list(range(4995,5000))


def test2_movements(img_num):
    class_num = 0
    if 0 <= img_num <= 60:
        class_num = pose_dictionary["normal"]
    elif 61 <= img_num <= 67:
        class_num = -1
    elif 68 <= img_num <= 146:
        class_num = pose_dictionary["square"]
    elif 147 <= img_num <= 152:
        class_num = -1
    elif 153 <= img_num <= 250:
        class_num = pose_dictionary["cross"]
    elif 251 <= img_num <= 254:
        class_num = -1
    elif 255 <= img_num <= 359:
        class_num = pose_dictionary["triangle"]
    elif 360 <= img_num <= 376:
        class_num = -1
    elif 377 <= img_num <= 509:
        class_num = pose_dictionary["circle"]
    elif 510 <= img_num <= 514:
        class_num = -1
    elif 515 <= img_num <= 541:
        class_num = pose_dictionary["normal"]
    elif 542 <= img_num <= 549:
        class_num = -1
    elif 550 <= img_num <= 610:
        class_num = pose_dictionary["square"]
    elif 611 <= img_num <= 619:
        class_num = -1
    elif 620 <= img_num <= 633:
        class_num = pose_dictionary["normal"]
    elif 634 <= img_num <= 639:
        class_num = -1
    elif 640 <= img_num <= 709:
        class_num = pose_dictionary["LT"]
    elif 710 <= img_num <= 715:
        class_num = -1
    elif 716 <= img_num <= 775:
        class_num = pose_dictionary["LB"]
    elif 776 <= img_num <= 780:
        class_num = -1
    elif 781 <= img_num <= 788:
        class_num = pose_dictionary["normal"]
    elif 789 <= img_num <= 797:
        class_num = -1
    elif 798 <= img_num <= 830:
        class_num = pose_dictionary["RT"]
    elif 831 <= img_num <= 835:
        class_num = -1
    elif 836 <= img_num <= 882:
        class_num = pose_dictionary["RB"]
    elif 883 <= img_num <= 887:
        class_num = -1
    elif 888 <= img_num <= 897:
        class_num = pose_dictionary["normal"]
    elif 898 <= img_num <= 908:
        class_num = -1
    elif 909 <= img_num <= 960:
        class_num = pose_dictionary["cross"]
    elif 961 <= img_num <= 968:
        class_num = -1
    elif 969 <= img_num <= 1010:
        class_num = pose_dictionary["normal"]
    elif 1011 <= img_num <= 1013:
        class_num = -1
    elif 1014 <= img_num <= 1095:
        class_num = pose_dictionary["L-up"]
    elif 1096 <= img_num <= 1102:
        class_num = -1
    elif 1103 <= img_num <= 1174:
        class_num = pose_dictionary["L-right"]
    elif 1175 <= img_num <= 1183:
        class_num = -1
    elif 1184 <= img_num <= 1194:
        class_num = pose_dictionary["normal"]
    elif 1195 <= img_num <= 1201:
        class_num = -1
    elif 1202 <= img_num <= 1317:
        class_num = pose_dictionary["L-left"]
    elif 1318 <= img_num <= 1324:
        class_num = -1
    elif 1325 <= img_num <= 1327:
        class_num = pose_dictionary["normal"]
    elif 1328 <= img_num <= 1335:
        class_num = -1
    elif 1336 <= img_num <= 1388:
        class_num = pose_dictionary["L-down"]
    elif 1389 <= img_num <= 1394:
        class_num = -1
    elif 1395 <= img_num <= 1409:
        class_num = pose_dictionary["normal"]
    elif 1410 <= img_num <= 1420:
        class_num = -1
    elif 1421 <= img_num <= 1471:
        class_num = pose_dictionary["triangle"]
    elif 1472 <= img_num <= 1481:
        class_num = -1
    elif 1482 <= img_num <= 1502:
        class_num = pose_dictionary["normal"]
    elif 1503 <= img_num <= 1507:
        class_num = -1
    elif 1508 <= img_num <= 1558:
        class_num = pose_dictionary["L-up"]
    elif 1559 <= img_num <= 1563:
        class_num = -1
    elif 1564 <= img_num <= 1582:
        class_num = pose_dictionary["normal"]
    elif 1583 <= img_num <= 1592:
        class_num = -1
    elif 1593 <= img_num <= 1702:
        class_num = pose_dictionary["R-up"]
    elif 1703 <= img_num <= 1726:
        class_num = -1
    elif 1727 <= img_num <= 1822:
        class_num = pose_dictionary["R-down"]
    elif 1823 <= img_num <= 1860:
        class_num = -1
    elif 1861 <= img_num <= 1967:
        class_num = pose_dictionary["R-right"]
    elif 1968 <= img_num <= 1988:
        class_num = -1
    elif 1989 <= img_num <= 2063:
        class_num = pose_dictionary["R-left"]
    elif 2064 <= img_num <= 2075:
        class_num = -1
    elif 2076 <= img_num <= 2101:
        class_num = pose_dictionary["normal"]
    elif 2102 <= img_num <= 2106:
        class_num = -1
    elif 2107 <= img_num <= 2203:
        class_num = pose_dictionary["up-arrow"]
    elif 2204 <= img_num <= 2213:
        class_num = -1
    elif 2214 <= img_num <= 2297:
        class_num = pose_dictionary["down-arrow"]
    elif 2298 <= img_num <= 2306:
        class_num = -1
    elif 2307 <= img_num <= 2389:
        class_num = pose_dictionary["right-arrow"]
    elif 2390 <= img_num <= 2402:
        class_num = -1
    elif 2403 <= img_num <= 2497:
        class_num = pose_dictionary["left-arrow"]
    elif 2498 <= img_num <= 2507:
        class_num = -1
    elif 2508 <= img_num <= 2523:
        class_num = pose_dictionary["normal"]
    elif 2524 <= img_num <= 2532:
        class_num = -1
    elif 2533 <= img_num <= 2582:
        class_num = pose_dictionary["circle"]
    elif 2583 <= img_num <= 2590:
        class_num = -1
    elif 2591 <= img_num <= 2601:
        class_num = pose_dictionary["normal"]
    elif 2602 <= img_num <= 2608:
        class_num = -1
    elif 2609 <= img_num <= 2647:
        class_num = pose_dictionary["RT"]
    elif 2648 <= img_num <= 2657:
        class_num = -1
    elif 2658 <= img_num <= 2681:
        class_num = pose_dictionary["normal"]
    elif 2682 <= img_num <= 2689:
        class_num = -1
    elif 2690 <= img_num <= 2821:
        class_num = pose_dictionary["praise"]
    elif 2822 <= img_num <= 2831:
        class_num = -1
    elif 2832 <= img_num <= 2865:
        class_num = pose_dictionary["normal"]
    elif 2866 <= img_num <= 2882:
        class_num = -1
    elif 2883 <= img_num <= 3017:
        class_num = pose_dictionary["bow"]
    elif 3018 <= img_num <= 3035:
        class_num = -1
    elif 3036 <= img_num <= 3050:
        class_num = pose_dictionary["normal"]
    elif 3051 <= img_num <= 3060:
        class_num = -1
    elif 3061 <= img_num <= 3145:
        class_num = pose_dictionary["crossarm"]
    elif 3146 <= img_num <= 3155:
        class_num = -1
    elif 3156 <= img_num <= 3171:
        class_num = pose_dictionary["normal"]
    elif 3172 <= img_num <= 3180:
        class_num = -1
    elif 3181 <= img_num <= 3223:
        class_num = pose_dictionary["cross"]
    elif 3224 <= img_num <= 3233:
        class_num = -1
    elif 3234 <= img_num <= 3272:
        class_num = pose_dictionary["normal"]
    elif 3273 <= img_num <= 3306:
        class_num = -1
    elif 3307 <= img_num <= 3552:
        class_num = pose_dictionary["options"]
    elif 3553 <= img_num <= 3585:
        class_num = -1
    elif 3586 <= img_num <= 3633:
        class_num = pose_dictionary["L-down"]
    elif 3634 <= img_num <= 3644:
        class_num = -1
    elif 3645 <= img_num <= 3709:
        class_num = pose_dictionary["normal"]
    elif 3710 <= img_num <= 3718:
        class_num = -1
    elif 3719 <= img_num <= 3802:
        class_num = pose_dictionary["square"]
    elif 3803 <= img_num <= 3812:
        class_num = -1
    elif 3813 <= img_num <= 3846:
        class_num = pose_dictionary["normal"]
    elif 3847 <= img_num <= 3857:
        class_num = -1
    elif 3858 <= img_num <= 3927:
        class_num = pose_dictionary["crossarm"]
    elif 3928 <= img_num <= 3937:
        class_num = -1
    elif 3938 <= img_num <= 3955:
        class_num = pose_dictionary["normal"]
    elif 3956 <= img_num <= 3960:
        class_num = -1
    elif 3961 <= img_num <= 4072:
        class_num = pose_dictionary["RB"]
    elif 4073 <= img_num <= 4076:
        class_num = -1
    elif 4077 <= img_num <= 4096:
        class_num = pose_dictionary["normal"]
    elif 4097 <= img_num <= 4101:
        class_num = -1
    elif 4102 <= img_num <= 4146:
        class_num = pose_dictionary["LT"]
    elif 4147 <= img_num <= 4155:
        class_num = -1
    elif 4156 <= img_num <= 4203:
        class_num = pose_dictionary["normal"]
    elif 4204 <= img_num <= 4208:
        class_num = -1
    elif 4209 <= img_num <= 4336:
        class_num = pose_dictionary["L-left"]
    elif 4337 <= img_num <= 4342:
        class_num = -1
    elif 4343 <= img_num <= 4395:
        class_num = pose_dictionary["normal"]
    elif 4396 <= img_num <= 4402:
        class_num = -1
    elif 4403 <= img_num <= 4503:
        class_num = pose_dictionary["up-arrow"]
    elif 4504 <= img_num <= 4513:
        class_num = -1
    elif 4514 <= img_num <= 4559:
        class_num = pose_dictionary["normal"]
    elif 4560 <= img_num <= 4566:
        class_num = -1
    elif 4567 <= img_num <= 4635:
        class_num = pose_dictionary["right-arrow"]
    elif 4636 <= img_num <= 4642:
        class_num = -1
    elif 4643 <= img_num <= 4655:
        class_num = pose_dictionary["normal"]
    elif 4656 <= img_num <= 4665:
        class_num = -1
    elif 4666 <= img_num <= 4742:
        class_num = pose_dictionary["R-left"]
    elif 4743 <= img_num <= 4751:
        class_num = -1
    elif 4752 <= img_num <= 4812:
        class_num = pose_dictionary["normal"]
    elif 4813 <= img_num <= 4818:
        class_num = -1
    elif 4819 <= img_num <= 4958:
        class_num = pose_dictionary["L-right"]
    elif 4959 <= img_num <= 4966:
        class_num = -1
    elif 4967 <= img_num <= 4994:
        class_num = pose_dictionary["normal"]
    elif 4995 <= img_num <= 4999:
        class_num = -1
    return class_num
