
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


fighting_dictionary = {
    "Defend": 0,
    "LeftKick": 1,
    "LeftPunch": 2,
    "Normal": 3,
    "RightKick": 4,
    "RightPunch": 5,
}

fight_pose_switches1 = list(range(58,62)) + list(range(99,102)) + list(range(117,123)) + list(range(151,155)) + \
                      list(range(183, 188)) + list(range(215, 229)) + list(range(235, 237)) + list(range(266,271)) + \
                      list(range(292, 297)) + list(range(343,352))


def fight_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 57:
        class_num = fighting_dictionary["Normal"]
    elif 58 <= img_num <= 61:
        class_num = -1
    elif 62 <= img_num <= 98:
        class_num = fighting_dictionary["RightPunch"]
    elif 99 <= img_num <= 101:
        class_num = -1
    elif 102 <= img_num <= 116:
        class_num = fighting_dictionary["Normal"]
    elif 117 <= img_num <= 122:
        class_num = -1
    elif 123 <= img_num <= 150:
        class_num = fighting_dictionary["LeftPunch"]
    elif 151 <= img_num <= 154:
        class_num = -1
    elif 155 <= img_num <= 182:
        class_num = fighting_dictionary["Normal"]
    elif 183 <= img_num <= 187:
        class_num = -1
    elif 188 <= img_num <= 214:
        class_num = fighting_dictionary["RightKick"]
    elif 215 <= img_num <= 228:
        class_num = -1
    elif 229 <= img_num <= 234:
        class_num = fighting_dictionary["Normal"]
    elif 235 <= img_num <= 236:
        class_num = -1
    elif 237 <= img_num <= 265:
        class_num = fighting_dictionary["LeftKick"]
    elif 266 <= img_num <= 270:
        class_num = -1
    elif 271 <= img_num <= 291:
        class_num = fighting_dictionary["Normal"]
    elif 292 <= img_num <= 296:
        class_num = -1
    elif 297 <= img_num <= 342:
        class_num = fighting_dictionary["Defend"]
    elif 343 <= img_num <= 351:
        class_num = -1
    elif 352 <= img_num <= 405:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches2 = list(range(80, 90)) + list(range(110,115)) + list(range(158, 167)) + list(range(183,186)) + \
                       list(range(242, 249)) + list(range(267,273)) + list(range(308, 313)) + list(range(330,337)) + \
                       list(range(371, 378)) + list(range(406, 414))


def fight_movements2(img_num):
    class_num = 0
    if 0 <= img_num <= 79:
        class_num = fighting_dictionary["Normal"]
    elif 80 <= img_num <= 89:
        class_num = -1
    elif 90 <= img_num <= 109:
        class_num = fighting_dictionary["RightPunch"]
    elif 110 <= img_num <= 114:
        class_num = -1
    elif 115 <= img_num <= 157:
        class_num = fighting_dictionary["Normal"]
    elif 158 <= img_num <= 166:
        class_num = -1
    elif 167 <= img_num <= 182:
        class_num = fighting_dictionary["LeftPunch"]
    elif 183 <= img_num <= 185:
        class_num = -1
    elif 186 <= img_num <= 241:
        class_num = fighting_dictionary["Normal"]
    elif 242 <= img_num <= 248:
        class_num = -1
    elif 249 <= img_num <= 266:
        class_num = fighting_dictionary["RightKick"]
    elif 267 <= img_num <= 272:
        class_num = -1
    elif 273 <= img_num <= 307:
        class_num = fighting_dictionary["Normal"]
    elif 308 <= img_num <= 312:
        class_num = -1
    elif 313 <= img_num <= 329:
        class_num = fighting_dictionary["LeftKick"]
    elif 330 <= img_num <= 336:
        class_num = -1
    elif 337 <= img_num <= 370:
        class_num = fighting_dictionary["Normal"]
    elif 371 <= img_num <= 377:
        class_num = -1
    elif 378 <= img_num <= 405:
        class_num = fighting_dictionary["Defend"]
    elif 406 <= img_num <= 413:
        class_num = -1
    elif 414 <= img_num <= 423:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches3 = list(range(88, 95)) + list(range(132, 137)) + list(range(158,165)) + list(range(198, 203)) +\
                       list(range(233,235)) + list(range(267, 272)) + list(range(312,314)) + list(range(350, 355)) + \
                       list(range(384, 391)) + list(range(454, 460))


def fight_movements3(img_num):
    class_num = 0
    if 0 <= img_num <= 87:
        class_num = fighting_dictionary["Normal"]
    elif 88 <= img_num <= 94:
        class_num = -1
    elif 95 <= img_num <= 131:
        class_num = fighting_dictionary["RightPunch"]
    elif 132 <= img_num <= 136:
        class_num = -1
    elif 137 <= img_num <= 157:
        class_num = fighting_dictionary["Normal"]
    elif 158 <= img_num <= 164:
        class_num = -1
    elif 165 <= img_num <= 197:
        class_num = fighting_dictionary["LeftPunch"]
    elif 198 <= img_num <= 202:
        class_num = -1
    elif 203 <= img_num <= 232:
        class_num = fighting_dictionary["Normal"]
    elif 233 <= img_num <= 234:
        class_num = -1
    elif 235 <= img_num <= 266:
        class_num = fighting_dictionary["RightKick"]
    elif 267 <= img_num <= 271:
        class_num = -1
    elif 272 <= img_num <= 311:
        class_num = fighting_dictionary["Normal"]
    elif 312 <= img_num <= 313:
        class_num = -1
    elif 314 <= img_num <= 349:
        class_num = fighting_dictionary["LeftKick"]
    elif 350 <= img_num <= 354:
        class_num = -1
    elif 355 <= img_num <= 383:
        class_num = fighting_dictionary["Normal"]
    elif 384 <= img_num <= 390:
        class_num = -1
    elif 391 <= img_num <= 453:
        class_num = fighting_dictionary["Defend"]
    elif 454 <= img_num <= 459:
        class_num = -1
    elif 460 <= img_num <= 482:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches4 = list(range(96, 100)) + list(range(117, 121)) + list(range(132, 138)) + list(range(198,204)) +\
                       list(range(228,231)) + list(range(253,255)) + list(range(285,288)) + list(range(333, 337)) + \
                       list(range(363,370))


def fight_movements4(img_num):
    class_num = 0
    if 0 <= img_num <= 95:
        class_num = fighting_dictionary["Normal"]
    elif 96 <= img_num <= 99:
        class_num = -1
    elif 100 <= img_num <= 116:
        class_num = fighting_dictionary["RightPunch"]
    elif 117 <= img_num <= 120:
        class_num = -1
    elif 121 <= img_num <= 131:
        class_num = fighting_dictionary["Normal"]
    elif 132 <= img_num <= 137:
        class_num = -1
    elif 138 <= img_num <= 162:
        class_num = fighting_dictionary["LeftPunch"]
    elif 163 <= img_num <= 197:
        class_num = fighting_dictionary["Normal"]
    elif 198 <= img_num <= 203:
        class_num = -1
    elif 204 <= img_num <= 227:
        class_num = fighting_dictionary["RightKick"]
    elif 228 <= img_num <= 230:
        class_num = -1
    elif 231 <= img_num <= 252:
        class_num = fighting_dictionary["Normal"]
    elif 253 <= img_num <= 254:
        class_num = -1
    elif 255 <= img_num <= 284:
        class_num = fighting_dictionary["LeftKick"]
    elif 285 <= img_num <= 288:
        class_num = -1
    elif 289 <= img_num <= 332:
        class_num = fighting_dictionary["Normal"]
    elif 333 <= img_num <= 336:
        class_num = -1
    elif 337 <= img_num <= 362:
        class_num = fighting_dictionary["Defend"]
    elif 363 <= img_num <= 369:
        class_num = -1
    elif 370 <= img_num <= 406:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches5 =  list(range(72, 76)) + list(range(108,114)) + list(range(136,138)) + list(range(171,174)) + \
                        list(range(225,227)) + list(range(257,261)) + list(range(304,306)) + list(range(337,340)) + \
                        list(range(373,381)) + list(range(411,420))


def fight_movements5(img_num):
    class_num = 0
    if 0 <= img_num <= 71:
        class_num = fighting_dictionary["Normal"]
    elif 72 <= img_num <= 75:
        class_num = -1
    elif 76 <= img_num <= 107:
        class_num = fighting_dictionary["RightPunch"]
    elif 108 <= img_num <= 113:
        class_num = -1
    elif 114 <= img_num <= 135:
        class_num = fighting_dictionary["Normal"]
    elif 136 <= img_num <= 137:
        class_num = -1
    elif 138 <= img_num <= 170:
        class_num = fighting_dictionary["LeftPunch"]
    elif 171 <= img_num <= 174:
        class_num = -1
    elif 175 <= img_num <= 224:
        class_num = fighting_dictionary["Normal"]
    elif 225 <= img_num <= 226:
        class_num = -1
    elif 227 <= img_num <= 256:
        class_num = fighting_dictionary["RightKick"]
    elif 257 <= img_num <= 261:
        class_num = -1
    elif 262 <= img_num <= 303:
        class_num = fighting_dictionary["Normal"]
    elif 304 <= img_num <= 305:
        class_num = -1
    elif 306 <= img_num <= 336:
        class_num = fighting_dictionary["LeftKick"]
    elif 337 <= img_num <= 339:
        class_num = -1
    elif 340 <= img_num <= 372:
        class_num = fighting_dictionary["Normal"]
    elif 373 <= img_num <= 380:
        class_num = -1
    elif 381 <= img_num <= 410:
        class_num = fighting_dictionary["Defend"]
    elif 411 <= img_num <= 419:
        class_num = -1
    elif 420 <= img_num <= 451:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches6 = list(range(67,72)) + list(range(110, 113)) + list(range(128, 134)) + list(range(163,166)) +\
                       list(range(163,201)) + list(range(224,229)) + list(range(265,269)) + list(range(291,304)) + \
                       list(range(327,335)) + list(range(362,369))


def fight_movements6(img_num):
    class_num = 0
    if 0 <= img_num <= 66:
        class_num = fighting_dictionary["Normal"]
    elif 67 <= img_num <= 71:
        class_num = -1
    elif 72 <= img_num <= 109:
        class_num = fighting_dictionary["RightPunch"]
    elif 110 <= img_num <= 112:
        class_num = -1
    elif 113 <= img_num <= 127:
        class_num = fighting_dictionary["Normal"]
    elif 128 <= img_num <= 134:
        class_num = -1
    elif 135 <= img_num <= 162:
        class_num = fighting_dictionary["LeftPunch"]
    elif 163 <= img_num <= 166:
        class_num = -1
    elif 167 <= img_num <= 195:
        class_num = fighting_dictionary["Normal"]
    elif 196 <= img_num <= 200:
        class_num = -1
    elif 201 <= img_num <= 223:
        class_num = fighting_dictionary["RightKick"]
    elif 224 <= img_num <= 228:
        class_num = -1
    elif 229 <= img_num <= 264:
        class_num = fighting_dictionary["Normal"]
    elif 265 <= img_num <= 268:
        class_num = -1
    elif 269 <= img_num <= 290:
        class_num = fighting_dictionary["LeftKick"]
    elif 291 <= img_num <= 303:
        class_num = -1
    elif 304 <= img_num <= 326:
        class_num = fighting_dictionary["Normal"]
    elif 327 <= img_num <= 334:
        class_num = -1
    elif 335 <= img_num <= 361:
        class_num = fighting_dictionary["Defend"]
    elif 362 <= img_num <= 369:
        class_num = -1
    elif 370 <= img_num <= 408:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches7 = list(range(81, 87)) + list(range(119,126)) + list(range(146,149)) + list(range(171,175)) + \
                       list(range(214,216)) + list(range(238,241)) + list(range(273,276)) + list(range(301,305)) +\
                       list(range(333,340)) + list(range(370,376))


def fight_movements7(img_num):
    class_num = 0
    if 0 <= img_num <= 80:
        class_num = fighting_dictionary["Normal"]
    elif 81 <= img_num <= 86:
        class_num = -1
    elif 87 <= img_num <= 118:
        class_num = fighting_dictionary["RightPunch"]
    elif 119 <= img_num <= 125:
        class_num = -1
    elif 126 <= img_num <= 145:
        class_num = fighting_dictionary["Normal"]
    elif 146 <= img_num <= 148:
        class_num = -1
    elif 149 <= img_num <= 170:
        class_num = fighting_dictionary["LeftPunch"]
    elif 171 <= img_num <= 174:
        class_num = -1
    elif 175 <= img_num <= 213:
        class_num = fighting_dictionary["Normal"]
    elif 214 <= img_num <= 215:
        class_num = -1
    elif 216 <= img_num <= 237:
        class_num = fighting_dictionary["RightKick"]
    elif 238 <= img_num <= 240:
        class_num = -1
    elif 241 <= img_num <= 272:
        class_num = fighting_dictionary["Normal"]
    elif 273 <= img_num <= 275:
        class_num = -1
    elif 276 <= img_num <= 300:
        class_num = fighting_dictionary["LeftKick"]
    elif 301 <= img_num <= 304:
        class_num = -1
    elif 305 <= img_num <= 332:
        class_num = fighting_dictionary["Normal"]
    elif 333 <= img_num <= 339:
        class_num = -1
    elif 340 <= img_num <= 369:
        class_num = fighting_dictionary["Defend"]
    elif 370 <= img_num <= 375:
        class_num = -1
    elif 376 <= img_num <= 415:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches8 = list(range(48,51)) + list(range(79,82)) + list(range(98,102)) + list(range(126,130)) + \
                       list(range(164,166)) + list(range(183,186)) + list(range(213,215)) + list(range(237,241)) + \
                       list(range(265,270)) + list(range(323, 334))


def fight_movements8(img_num):
    class_num = 0
    if 0 <= img_num <= 47:
        class_num = fighting_dictionary["Normal"]
    elif 48 <= img_num <= 50:
        class_num = -1
    elif 51 <= img_num <= 78:
        class_num = fighting_dictionary["RightPunch"]
    elif 79 <= img_num <= 81:
        class_num = -1
    elif 82 <= img_num <= 97:
        class_num = fighting_dictionary["Normal"]
    elif 98 <= img_num <= 101:
        class_num = -1
    elif 102 <= img_num <= 125:
        class_num = fighting_dictionary["LeftPunch"]
    elif 126 <= img_num <= 129:
        class_num = -1
    elif 130 <= img_num <= 163:
        class_num = fighting_dictionary["Normal"]
    elif 164 <= img_num <= 165:
        class_num = -1
    elif 166 <= img_num <= 182:
        class_num = fighting_dictionary["RightKick"]
    elif 183 <= img_num <= 185:
        class_num = -1
    elif 186 <= img_num <= 212:
        class_num = fighting_dictionary["Normal"]
    elif 213 <= img_num <= 214:
        class_num = -1
    elif 215 <= img_num <= 236:
        class_num = fighting_dictionary["LeftKick"]
    elif 237 <= img_num <= 240:
        class_num = -1
    elif 241 <= img_num <= 264:
        class_num = fighting_dictionary["Normal"]
    elif 265 <= img_num <= 269:
        class_num = -1
    elif 270 <= img_num <= 322:
        class_num = fighting_dictionary["Defend"]
    elif 323 <= img_num <= 333:
        class_num = -1
    elif 334 <= img_num <= 350:
        class_num = fighting_dictionary["Normal"]
    return class_num


fight_pose_switches9 = list(range(67, 76)) + list(range(129,137)) + list(range(164, 169)) + list(range(208, 215)) + \
                       list(range(262,264)) + list(range(301,305)) + list(range(348,350)) + list(range(395,402)) +\
                       list(range(447, 456)) + list(range(489,500))


def fight_movements9(img_num):
    class_num = 0
    if 0 <= img_num <= 66:
        class_num = fighting_dictionary["Normal"]
    elif 67 <= img_num <= 75:
        class_num = -1
    elif 76 <= img_num <= 128:
        class_num = fighting_dictionary["RightPunch"]
    elif 129 <= img_num <= 136:
        class_num = -1
    elif 137 <= img_num <= 163:
        class_num = fighting_dictionary["Normal"]
    elif 164 <= img_num <= 168:
        class_num = -1
    elif 169 <= img_num <= 207:
        class_num = fighting_dictionary["LeftPunch"]
    elif 208 <= img_num <= 214:
        class_num = -1
    elif 215 <= img_num <= 261:
        class_num = fighting_dictionary["Normal"]
    elif 262 <= img_num <= 263:
        class_num = -1
    elif 264 <= img_num <= 300:
        class_num = fighting_dictionary["RightKick"]
    elif 301 <= img_num <= 304:
        class_num = -1
    elif 305 <= img_num <= 347:
        class_num = fighting_dictionary["Normal"]
    elif 348 <= img_num <= 349:
        class_num = -1
    elif 350 <= img_num <= 394:
        class_num = fighting_dictionary["LeftKick"]
    elif 395 <= img_num <= 401:
        class_num = -1
    elif 402 <= img_num <= 446:
        class_num = fighting_dictionary["Normal"]
    elif 447 <= img_num <= 455:
        class_num = -1
    elif 456 <= img_num <= 488:
        class_num = fighting_dictionary["Defend"]
    elif 489 <= img_num <= 499:
        class_num = -1
    elif 500 <= img_num <= 536:
        class_num =fighting_dictionary["Normal"]
    return class_num


fight_pose_switches10 = list(range(231,236)) + list(range(256,259)) + list(range(294,299)) + list(range(319,323)) + \
                        list(range(364,366)) + list(range(388,392)) + list(range(436,438)) + list(range(462,467)) + \
                        list(range(496,503)) + list(range(529,539))


def fight_movements10(img_num):
    class_num = 0
    if 0 <= img_num <= 230:
        class_num = fighting_dictionary["Normal"]
    elif 231 <= img_num <= 235:
        class_num = -1
    elif 236 <= img_num <= 255:
        class_num = fighting_dictionary["RightPunch"]
    elif 256 <= img_num <= 258:
        class_num = -1
    elif 259 <= img_num <= 293:
        class_num = fighting_dictionary["Normal"]
    elif 294 <= img_num <= 298:
        class_num = -1
    elif 299 <= img_num <= 318:
        class_num = fighting_dictionary["LeftPunch"]
    elif 319 <= img_num <= 322:
        class_num = -1
    elif 323 <= img_num <= 363:
        class_num = fighting_dictionary["Normal"]
    elif 364 <= img_num <= 365:
        class_num = -1
    elif 366 <= img_num <= 387:
        class_num = fighting_dictionary["RightKick"]
    elif 388 <= img_num <= 391:
        class_num = -1
    elif 392 <= img_num <= 435:
        class_num = fighting_dictionary["Normal"]
    elif 436 <= img_num <= 437:
        class_num = -1
    elif 438 <= img_num <= 461:
        class_num = fighting_dictionary["LeftKick"]
    elif 462 <= img_num <= 466:
        class_num = -1
    elif 467 <= img_num <= 495:
        class_num = fighting_dictionary["Normal"]
    elif 496 <= img_num <= 502:
        class_num = -1
    elif 503 <= img_num <= 528:
        class_num = fighting_dictionary["Defend"]
    elif 529 <= img_num <= 538:
        class_num = -1
    elif 539 <= img_num <= 643:
        class_num = fighting_dictionary["Normal"]
    return class_num


golf_dictionary = {
    "Golf": 0,
    "Normal": 1
}

golf_pose_switches1 = list(range(63,73)) + list(range(156,178))


def golf_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 62:
        class_num = golf_dictionary["Normal"]
    elif 63 <= img_num <= 72:
        class_num = -1
    elif 73 <= img_num <= 155:
        class_num = golf_dictionary["Golf"]
    elif 156 <= img_num <= 177:
        class_num = -1
    elif 178 <= img_num <= 223:
        class_num = golf_dictionary["Normal"]
    return class_num


tennis_dictionary = {
    "Backhand": 0,
    "Forehand": 1,
    "Normal": 2,
    "TennisServe": 3
}

tennis_pose_switches1 = list(range(69,77)) + list(range(132,142)) + list(range(192,197)) + list(range(246,255)) +\
                        list(range(307,315)) + list(range(377,380))


def tennis_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 68:
        class_num = tennis_dictionary["Normal"]
    elif 69 <= img_num <= 76:
        class_num = -1
    elif 77 <= img_num <= 131:
        class_num = tennis_dictionary["Forehand"]
    elif 132 <= img_num <= 141:
        class_num = -1
    elif 142 <= img_num <= 191:
        class_num = tennis_dictionary["Normal"]
    elif 192 <= img_num <= 196:
        class_num = -1
    elif 197 <= img_num <= 245:
        class_num = tennis_dictionary["Backhand"]
    elif 246 <= img_num <= 254:
        class_num = -1
    elif 255 <= img_num <= 306:
        class_num = tennis_dictionary["Normal"]
    elif 307 <= img_num <= 314:
        class_num = -1
    elif 315 <= img_num <= 376:
        class_num = tennis_dictionary["TennisServe"]
    elif 377 <= img_num <= 379:
        class_num = -1
    elif 380 <= img_num <= 428:
        class_num = tennis_dictionary["Normal"]
    return class_num


bowling_dictionary = {
    "Normal": 0,
    "ThrowBall": 1
}


bowling_pose_switches1 = list(range(57,70)) + list(range(168,178))


def bowling_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 56:
        class_num = bowling_dictionary["Normal"]
    elif 57 <= img_num <= 69:
        class_num = -1
    elif 70 <= img_num <= 167:
        class_num = bowling_dictionary["ThrowBall"]
    elif 168 <= img_num <= 177:
        class_num = -1
    elif 178 <= img_num <= 205:
        class_num = bowling_dictionary["Normal"]
    return class_num


fps_dictionary = {
    "AimCenter": 0,
    "AimLeft": 1,
    "AimRight": 2,
    "Climb": 3,
    "Crouch": 4,
    "Jump": 5,
    "Normal": 6,
    "Run": 7,
    "Walk": 8
}


fps_pose_switches1 = list(range(46,51)) + list(range(249,257)) + list(range(477,482)) + list(range(563,546)) + list(range(585, 596))


def fps_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 45:
        class_num = fps_dictionary["Normal"]
    elif 46 <= img_num <= 50:
        class_num = -1
    elif 51 <= img_num <= 121:
        class_num = fps_dictionary["AimCenter"]
    elif 122 <= img_num <= 191:
        class_num = fps_dictionary["AimRight"]
    elif 192 <= img_num <= 196:
        class_num = fps_dictionary["AimCenter"]
    elif 197 <= img_num <= 248:
        class_num = fps_dictionary["AimLeft"]
    elif 249 <= img_num <= 256:
        class_num = -1
    elif 257 <= img_num <= 285:
        class_num = fps_dictionary["Normal"]
    elif 286 <= img_num <= 355:
        class_num = fps_dictionary["Walk"]
    elif 356 <= img_num <= 417:
        class_num = fps_dictionary["Run"]
    elif 418 <= img_num <= 421:
        class_num = fps_dictionary["Normal"]
    elif 422 <= img_num <= 445:
        class_num = fps_dictionary["Jump"]
    elif 446 <= img_num <= 476:
        class_num = fps_dictionary["Normal"]
    elif 477 <= img_num <= 481:
        class_num = -1
    elif 482 <= img_num <= 535:
        class_num = fps_dictionary["Climb"]
    elif 536 <= img_num <= 545:
        class_num = -1
    elif 546 <= img_num <= 584:
        class_num = fps_dictionary["Crouch"]
    elif 585 <= img_num <= 595:
        class_num = -1
    elif 596 <= img_num <= 617:
        class_num = fps_dictionary["Normal"]
    return class_num


driving_dictionary = {
    "HoldWheel": 0,
    "Normal": 1,
    "TurnLeft": 2,
    "TurnRight": 3
}


driving_pose_switches1 = list(range(77,84)) + list(range(377, 382))


def driving_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 76:
        class_num = driving_dictionary["Normal"]
    elif 77 <= img_num <= 83:
        class_num = -1
    elif 84 <= img_num <= 159:
        class_num = driving_dictionary["HoldWheel"]
    elif 160 <= img_num <= 228:
        class_num = driving_dictionary["TurnRight"]
    elif 229 <= img_num <= 263:
        class_num = driving_dictionary["HoldWheel"]
    elif 264 <= img_num <= 314:
        class_num = driving_dictionary["TurnLeft"]
    elif 315 <= img_num <= 376:
        class_num = driving_dictionary["HoldWheel"]
    elif 377 <= img_num <= 381:
        class_num = -1
    elif 382 <= img_num <= 436:
        class_num = driving_dictionary["Normal"]
    return class_num


misc_dictionary = {
    "Clap": 0,
    "FlapArms": 1,
    "Normal": 2,
    "Wave": 3
}


misc_pose_switches1 = list(range(6,12)) + list(range(103,107)) + list(range(191,195)) + list(range(252,256))


def misc_movements1(img_num):
    class_num = 0
    if 0 <= img_num <= 5:
        class_num = misc_dictionary["Normal"]
    elif 6 <= img_num <= 11:
        class_num = -1
    elif 12 <= img_num <= 102:
        class_num = misc_dictionary["Wave"]
    elif 103 <= img_num <= 106:
        class_num = -1
    elif 107 <= img_num <= 113:
        class_num = misc_dictionary["Normal"]
    elif 114 <= img_num <= 190:
        class_num = misc_dictionary["FlapArms"]
    elif 191 <= img_num <= 194:
        class_num = -1
    elif 195 <= img_num <= 251:
        class_num = misc_dictionary["Clap"]
    elif 252 <= img_num <= 255:
        class_num = -1
    elif 256 <= img_num <= 293:
        class_num = misc_dictionary["Normal"]
    return class_num



# Code to fix tests file names
"""
import os
def rename_images():
    full_path = "gaming_dataset/tennis/test10-tennis/"
    img_num = 0
    colour_num = 1
    cur_img = full_path+f"Colour {colour_num}.png"

    while img_num < 313:
        if os.path.isfile(cur_img):
            os.rename(cur_img, full_path+f"img{img_num}.png")
            img_num += 1
        colour_num += 1
        cur_img = full_path+f"Colour {colour_num}.png"


if __name__ == '__main__':
    rename_images()
"""
