#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script extracts the unique set of distinct labels across languages (focusing for the moment on English and German only)
    Created on Mon Mar 5 2018

    @author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)
"""

import json
import cPickle as pkl
def process_json(file_):
    # Process the json data elements
    print("Processing json file ...")
    with open(file_) as data_file:
        data = json.load(data_file)
    return data

def generate_dict_pkl():
    json_path_dev = "/aimlx/mhan/data/dw_general/dev/"
    json_path_train = "/aimlx/mhan/data/dw_general/train/"
    json_path_test = "/aimlx/mhan/data/dw_general/test/"
    """ Mapping list of ids in English to English unique ids"""
    en_train = process_json(json_path_train+"english.json")
    en_dev = process_json(json_path_dev+"english.json")
    en_test = process_json(json_path_test+"english.json")

    eny = en_train["Y_ids"] + en_dev["Y_ids"] + en_test["Y_ids"]

    en_y = set(tuple(row) for row in eny)

    print(en_y)

    dict_ids_en = {}
    i = 0
    for y in en_y:
        dict_ids_en.update({y:i})
        i = i + 1

    print("dict_ids_en = ",dict_ids_en)

    dict_ids_en_rev = {}
    i = 0
    for y in en_y:
        dict_ids_en_rev.update({i:y})
        i = i + 1


    """ Mapping list of ids in German to German unique ids"""
    de_train = process_json(json_path_train+"german.json")
    de_dev = process_json(json_path_dev+"german.json")
    de_test = process_json(json_path_test+"german.json")

    dey = de_train["Y_ids"] + de_dev["Y_ids"] + de_test["Y_ids"]
    de_y = set(tuple(row) for row in dey)

    dict_ids_de = {}
    i = 0
    for y in de_y:
        dict_ids_de.update({y:i})
        i = i + 1

    print("dict_ids_de = ",dict_ids_de)

    dict_ids_de_rev = {}
    i = 0
    for y in de_y:
        dict_ids_de_rev.update({i:y})
        i = i + 1

    """ Mapping list of ids in English to list of ids in German"""
    dict_ids = {0: 199, 1: 362, 2: 245, 3: 184, 5: 22, 6: 254, 7: 147, 8: 67, 11: 216, 12: 207, 13: 24, 14: 223, 15: 159,
                16: 221, 18: 309, 20: 52, 26: 69, 27: 29, 29: 8, 31: 86, 34: 208, 37: 316, 38: 247, 39: 104, 40: 211, 41: 24,
                44: 203, 45: 359, 47: 94, 48: 185, 51: 315, 52: 155, 54: 92, 56: 278, 59: 116, 62: 66, 63: 192, 64: 145, 65: 298,
                67: 250, 68: 282, 71: 29, 72: 146, 74: 334, 75: 289, 76: 363, 80: 267, 82: 240, 87: 88, 92: 71, 98: 255, 99: 143,
                101: 321, 103: 35, 108: 89, 109: 81, 110: 275, 111: 37, 112: 283, 114: 349, 115: 326, 117: 114, 118: 13, 120: 166,
                124: 183, 125: 332, 127: 76, 128: 111, 131: 141, 132: 327, 135: 307, 137: 120, 139: 303, 140: 63, 142: 304, 143: 39,
                144: 204, 147: 231, 152: 296, 153: 274, 154: 261, 155: 285, 157: 139, 158: 133, 159: 233, 160: 248, 161: 337, 162: 346,
                165: 341, 166: 207, 167: 224, 169: 173, 170: 318, 173: 175, 176: 6, 178: 7, 183: 313, 187: 335, 188: 269, 189: 107,
                190: 305, 197: 291, 199: 340, 200: 347, 204: 42, 205: 270, 207: 171, 212: 356, 216: 325, 217: 4, 218: 246, 219: 364,
                220: 343, 223: 16, 224: 34, 225: 246, 227: 64, 228: 124, 229: 87, 232: 1, 233: 319, 236: 276, 237: 305, 244: 315,
                245: 118, 246: 249, 248: 117, 249: 202, 250: 80, 253: 85, 254: 239, 258: 354, 264: 25, 265: 297, 267: 294, 268: 328,
                269: 201, 270: 6, 272: 77, 275: 0, 276: 257, 279: 84, 280: 153, 282: 27, 283: 234, 287: 268, 288: 103, 290: 43, 291: 200,
                292: 174, 298: 93, 299: 264, 301: 122, 304: 219, 306: 281, 308: 17, 309: 188, 310: 355, 312: 123, 314: 331, 316: 263, 317: 152,
                319: 180, 321: 252, 324: 109, 325: 366}


    with open("dict_ids_en.p","wb") as dict_pkl:
        pkl.dump(dict_ids_en,dict_pkl)

    with open("dict_ids_en_rev.p","wb") as dict_pkl:
        pkl.dump(dict_ids_en_rev,dict_pkl)

    with open("dict_ids_de.p","wb") as dict_pkl:
        pkl.dump(dict_ids_de,dict_pkl)

    with open("dict_ids_de_rev.p","wb") as dict_pkl:
        pkl.dump(dict_ids_de_rev,dict_pkl)

    with open("dict_ids.p","wb") as dict_pkl:
        pkl.dump(dict_ids,dict_pkl)


    labels_map = {'world champions': 'weltmeister', 'world war i': 'erster weltkrieg', 'communication': 'kommunikation', 'special': 'special', 'angola': 'angola', 'wagner 200': 'wagner 200', 'world': 'welt', 'photo gallery': 'bildergalerie', 'federal election - podcast': 'bundestagswahl - podcast', '60 years dw': '60 jahre deutsche welle', 'plan b': 'plan b', 'exit': 'exit', 'economic regions': 'deutschlands wirtschaftsregionen', 'tanzania': 'tansania', 'digital culture': 'netzkultur', 'portugal': 'portugal', 'more sports': 'sportarten', 'the hungry world': 'die hungrige welt', 'the bundesliga': 'bundesliga', 'startups berlin': 'start-ups berlin', 'healthy living': 'gesund leben', 'travel': 'reise', 'quadriga': 'quadriga', 'germany': 'deutschland', 'video on demand': 'video on demand', 'weblog awards': 'auszeichnungen', 'in good shape': 'fit & gesund', 'kick countdown': 'kick countdown', '1945 - a turning point for europe': 'deutschland seit 1945', 'family business': 'familienunternehmen', 'lamb and game': 'lamm und wild', 'politics and society': 'politik & gesellschaft', 'world cup 2014': 'wm 2014', '125 years of cars - video': '125 jahre auto - video', 'destination germany': 'auf dem durch deutschland', 'pop': 'pop', 'the new arab debates': 'umbruch in der arabischen welt', 'strangers': 'fremde heimat', 'digital': 'digital', 'asia': 'asien', 'unrest in north africa': 'umbruch in nordafrika', 'americas': 'amerika', 'check-in': 'check-in', 'vegetarian dishes': 'vegetarische gerichte', 'space': 'weltraum', 'my': 'mein', 'middle east': 'nahost', 'secrets of transformation': 'secrets of transformation', 'kick it like': 'kick it like', 'videos': 'videos', 'society': 'gesellschaft', 'arab world economy': 'arabische welt', 'my favorite': 'mein favorit', 'brave tern - working on the edge': 'brave tern \\u2013 arbeiten am limit', 'reporters': 'reporter', 'usa': 'usa', 'on tour': 'deutschlandtour', 'south africa': 's\xc3\xbcdafrika', 'energy': 'energie', 'a trip across germany': 'deutschlandreise', 'football made in germany': 'football made in germany', 'fall of the wall': '25 jahre mauerfall', 'germany in 7 days': 'deutschland in 7 tagen', 'china': 'china', 'south 20 years after apartheid': 's\xc3\xbcdafrika 20 jahre nach der apartheid', 'israel': 'israel', '25 years german unification': '25 jahre deutsche einheit', 'our experts': 'unsere experten', 'ukraine': 'ukraine', 'human rights': 'menschenrechte', 'world health summit': 'world health summit', 'the storytellers': 'geschichte', 'faith matters': 'glaubenssachen', 'spain': 'spanien', 'education for all': 'bildung f\xc3\xbcr alle', 'the truth about germany': 'die Wahrheit \xc3\xbcber Deutschland', 'question of the week': 'frage der woche', 'study in germany': 'studieren in deutschland', 'bundesliga kick': 'bundesliga kick', 'nigeria': 'nigeria', 'awards': 'auszeichnungen', 'arts': 'kunst', 'economy and environment': 'wissen & umwelt', 'turkey': 't\xc3\xbcrkei', 'terra incognita': 'terra incognita', 'destination europe': 'flucht nach europa', 'young global leaders \\u2013 india start-up': 'young global leaders - neustart indien', 'nobel prize winners in lindau': 'nobelpreistr\xc3\xa4ger in lindau', 'dw news': 'dw nachrichten', 'reporter': 'reporter', 'podcast': 'podcast', 'europe on the edge': 'europa am rande', 'shift videos': 'shift videos', 'guinea-conakry': 'guinea-conakry', 'healthy eating': 'fit & gesund', 'kick': 'kick', 'nato summit chicago 2012': 'der nato-gipfel in chicago', 'made in germany': 'made in germany', 'highlights': 'highlights', 'germany today': 'deutschland heute', 'the car making world': 'autowelt', 'rice and pasta': 'reis und nudeln', 'strategy': 'strategie', 'olympics': 'olympia', 'culture': 'kultur', 'film': 'filme', 'archive': 'archiv', 'books': 'b\xc3\xbccher', 'climate change': 'gesichter des klimawandels', 'interview': 'interview', 'your opinion': 'ihre meinung', 'germany by scooter': 'auf dem durch deutschland', 'argentina': 'argentinien', 'fish and seafood': 'fisch und meeresfr\xc3\xbcchte', 'dtm': 'dtm', 'forum': 'forum', 'medical research': 'medizinforschung', 'col\\xf3n ring': 'der col\xc3\xb3n ring', 'faces of germany': 'gesichter deutschlands', 'poultry': 'gefl\xc3\xbcgel', 'global ideas': 'global ideas', 'dream destination - bundestag': 'traumziel bundestag', 'worldwide berlin': 'worldwide berlin', 'brazil': 'brasilien', 'questionnaire': 'fragebogen', 'mozambique': 'mosambik', 'wind power -- a new course for heligoland': 'windkraft \\u2013 neuer kurs f\xc3\xbcr helgoland', 'kino': 'kino', 'faces of climate change': 'gesichter des klimawandels', 'the strange world of max': 'die wundersame welt des max', 'africa': 'afrika', 'mobility': 'mobilit\xc3\xa4t', 'beef and pork': 'rind und schwein', 'jigsaw puzzle': 'puzzle', 'generation 25': 'generation 25', '- youtube': '- youtube', 'portrait': 'portrait', 'final': 'wm-finale', 'my 2030': 'mein 2030', 'those germans': 'so ticken die deutschen', 'topics': 'themen und dossiers', 'breaking news': 'nachrichten', 'lithuania': 'litauen', 'schumann at': 'schumann at', 'download': 'download', 'sports': 'sport', 'power to the people': 'alle macht dem', 'shift': 'shift', 'focus on europe': 'fokus europa', 'ireland': 'irland', 'india and friends and foes': 'indien und rivalisierende partner', 'news': 'nachrichten', 'desserts and sweet temptations': 'desserts und s\xc3\xbc\xc3\x9fe Gerichte', 'beethovenfest': 'beethovenfest', 'music': 'musik', 'bundesliga': 'bundesliga', 'expedition': 'expedition', '\\xe0 la carte': '\\xe0 la carte', 'a': 'a', 'facts about germany': 'deutschland verstehen', 'global 3000': 'global 3000', 'educational capital': 'bildung', 'kick special report': 'die kick reportage', 'podcasts': 'podcasts', 'managers of tomorrow': 'manager von morgen', 'greece': 'griechenland'}
    with open("labels_map.p","wb") as label_pkl:
        pkl.dump(labels_map,label_pkl)

    # dict_lists_en_de = {[3, 271]: [258], [284, 284, 47]: [298, 78, 190, 126], [32]: [317, 97], [86]: [165, 28], [225]: [189, 4], [280]: [147, 194], [65, 269]: [184], [175, 222]: [298, 144], [12]: [30, 59], [293, 56]: [234, 289], [79, 76, 195]: [337, 237, 41, 338], [82, 295]: [234, 237, 295], [283, 238, 218, 19]: [147, 363, 291], [188, 271]: [298, 352], [65, 21, 26]: [337, 237, 41, 284], [65, 326]: [234, 237, 31], [161, 289]: [240, 248], [103]: [298, 77, 210], [24]: [246], [65, 87, 173]: [337, 141], [79, 133]: [337, 237, 41], [103, 271]: [298, 145], [79, 76]: [189, 75], [0]: [234, 237, 308], [65, 22]: [298, 221, 48, 87], [293, 121]: [199, 152], [0, 271]: [268, 237, 320], [284]: [189, 168], [163, 234, 5, 164]: [165, 206], [65, 21, 165]: [268, 187], [82, 14]: [298, 78, 190, 129], [283, 142, 221, 5]: [82, 244], [65, 21, 109]: [337, 85], [306, 56]: [82, 67], [306, 295]: [40], [284, 157]: [147, 138], [163, 234, 5, 7]: [30, 245], [36]: [176, 276], [287, 153]: [298, 237, 177], [65, 272, 128]: [234, 237, 100], [65, 282, 1]: [268], [246, 271]: [207, 86], [83]: [240, 134], [68, 213, 120]: [174], [172, 35]: [298, 77, 340], [305, 271]: [30], [306, 151]: [147, 363], [314]: [113], [82, 162]: [268, 12], [283, 238]: [234, 237, 318], [68, 249]: [207], [175, 52]: [82, 143], [163]: [234, 237, 158], [283, 238, 218]: [240], [305, 199]: [298, 77, 287], [287, 205]: [51], [37, 245]: [70], [161, 148, 61]: [147], [79, 76, 96]: [189, 288], [68, 144]: [298, 221, 48, 356], [323, 315]: [38, 348], [25]: [298, 237, 56], [79]: [189, 213, 166], [68, 217, 165]: [176, 315], [163, 234, 101]: [268, 237, 106], [310]: [298, 27, 117], [161, 148, 130]: [300, 355], [283, 142, 107]: [234, 196], [82, 239]: [339], [65, 87, 308]: [298, 237, 314], [306, 106]: [298, 27, 362, 282], [79, 76, 209]: [82, 328], [284, 271]: [343], [156, 116]: [189, 71], [235]: [176], [293, 8]: [268, 24], [65, 282, 1, 68]: [25, 230], [172, 58]: [298, 365], [71]: [53], [11]: [298, 88, 294], [283, 34]: [152], [65, 282, 303]: [298, 237, 79], [33]: [128, 332], [302]: [337, 337], [283, 55, 206]: [234, 237, 161], [287, 41]: [268, 237], [284, 241]: [189, 45], [68, 294]: [337, 156], [68, 213, 115]: [189], [65, 282, 66, 316]: [337, 237], [82, 152]: [317, 292], [296, 287]: [234, 237, 242], [68, 16]: [317, 191], [319]: [128, 238], [82, 240]: [298, 77, 229], [283, 238, 247, 277]: [147, 138, 322], [283, 126]: [234, 304], [283, 238, 218, 278]: [298, 27, 362, 263], [65, 203]: [317, 108], [79, 176]: [82, 7], [172, 2]: [147, 82], [79, 76, 250]: [280], [65, 21, 207]: [249], [65, 272, 179]: [240, 361], [79, 76, 159]: [268, 237, 170, 270], [163, 234, 95]: [317], [65, 87, 98]: [298, 237], [161, 154]: [30, 226], [194]: [298], [163, 234]: [189, 259], [175, 53]: [147, 127], [79, 76, 270]: [317, 228], [163, 234, 101, 259]: [30, 286], [284, 202]: [311], [163, 125]: [298, 221, 48, 8], [261]: [298, 88, 17], [293, 79]: [360], [192, 170]: [298, 359, 5], [81]: [268, 182], [302, 271]: [189, 309], [79, 76, 307]: [354], [65, 297]: [30, 83], [65, 21]: [215], [79, 76, 127]: [151], [306, 67]: [298, 221, 319], [161]: [298, 27], [60]: [234, 2], [215]: [102], [293, 268]: [147, 363, 163], [79, 76, 39]: [298, 27, 112], [190]: [176, 285], [12, 166]: [147, 363, 233], [10]: [90, 148], [65, 232]: [82, 185], [161, 131]: [35], [68, 217]: [132], [65, 21, 29]: [234, 73], [163, 234, 101, 191]: [165, 188], [216]: [234, 237, 149], [305, 208]: [240, 358], [283, 75]: [82, 350], [161, 255]: [165, 20], [90]: [234, 237, 158, 110], [12, 271]: [300, 92], [283, 238, 218, 73]: [298, 88, 175], [27]: [298, 77, 64], [287, 146]: [35, 43], [163, 118]: [281], [220]: [82, 163], [319, 4]: [298, 203], [12, 105]: [325, 34], [296]: [337, 337, 65], [94]: [298, 26], [283, 38]: [18, 287], [65, 87, 80]: [317, 306], [55]: [317, 345], [82, 93]: [337, 269], [262]: [165, 139], [68, 15]: [298, 311], [305, 227]: [234, 237, 11], [82]: [240, 290], [283, 142, 122]: [234, 74], [283, 233]: [297], [65, 272, 119]: [234, 37, 137], [172, 99]: [234],[283, 142, 221]: [281, 64]}

    # dict_lists_de_en = {[30, 59]: [12], [298, 77, 229]: [82, 240], [82, 163]: [220], [234, 237, 295]: [82, 295], [280]: [79, 76, 250], [298, 144]: [175, 222], [317]: [163, 234, 95], [30, 226]: [161, 154], [298, 237, 314]: [65, 87, 308], [298, 88, 294]: [11], [82, 350]: [283, 75], [240, 290]: [82], [343]: [284, 271], [234, 289]: [293, 56], [298, 221, 48, 356]: [68, 144], [337, 141]: [65, 87, 173], [317, 228]: [79, 76, 270], [300, 92]: [12, 271], [298, 365]: [172, 58], [82, 244]: [283, 142, 221, 5], [234, 37, 137]: [65, 272, 119], [53]: [71], [246]: [24], [337, 237, 41]: [79, 133], [360]: [293, 79], [176, 315]: [68, 217, 165], [297]: [283, 233], [189, 75]: [79, 76], [147, 363, 233]: [12, 166], [234]: [172, 99], [268, 237, 320]: [0, 271], [298, 78, 190, 129]: [82, 14], [298, 359, 5]: [192, 170], [298, 26]: [94], [189, 4]: [225], [184]: [65, 269], [339]: [82, 239], [298, 237]: [65, 87, 98], [298, 221, 319]: [306, 67], [234, 237, 149]: [216], [337, 237, 41, 284]: [65, 21, 26], [234, 237, 31]: [65, 326], [128, 238]: [319], [268, 182]: [81], [147, 363, 163]: [293, 268], [298, 221, 48, 8]: [163, 125], [176, 285]: [190], [35, 43]: [287, 146], [189, 288]: [79, 76, 96], [298, 237, 177]: [287, 153], [165, 206]: [163, 234, 5, 164], [18, 287]: [283, 38], [268, 237, 170, 270]: [79, 76, 159], [176]: [235], [234, 237, 100]: [65, 272, 128], [189, 45]: [284, 241], [113]: [314], [337, 237]: [65, 282, 66, 316], [268, 237, 106]: [163, 234, 101], [317, 306]: [65, 87, 80], [337, 337, 65]: [296], [298, 145]: [103, 271], [234, 237, 242]: [296, 287], [234, 237, 318]: [283, 238], [90, 148]: [10], [82, 185]: [65, 232], [240, 134]: [83], [147, 82]: [172, 2], [189, 213, 166]: [79], [298, 77, 340]: [172, 35], [234, 196]: [283, 142, 107], [298, 88, 17]: [261], [165, 139]: [262], [38, 348]: [323, 315], [298, 78, 190, 126]: [284, 284, 47], [147, 194]: [280], [317, 97]: [32], [281, 64]: [283, 142, 221], [189, 309]: [302, 271], [234, 237, 158]: [163], [30, 286]: [163, 234, 101, 259], [298, 352]: [188, 271], [268]: [65, 282, 1], [147]: [161, 148, 61], [147, 363]: [306, 151], [300, 355]: [161, 148, 130], [234, 237, 161]: [283, 55, 206], [234, 74]: [283, 142, 122], [298, 27, 362, 282]: [306, 106], [40]: [306, 295], [30, 245]: [163, 234, 5, 7], [240, 358]: [305, 208], [151]: [79, 76, 127], [35]: [161, 131], [189]: [68, 213, 115], [176, 276]: [36], [25, 230]: [65, 282, 1, 68], [207, 86]: [246, 271], [128, 332]: [33], [298, 237, 56]: [25], [298, 27, 117]: [310], [165, 20]: [161, 255], [240]: [283, 238, 218], [70]: [37, 245], [234, 237, 11]: [305, 227], [82, 328]: [79, 76, 209], [165, 188]: [163, 234, 101, 191], [152]: [283, 34], [51]: [287, 205], [337, 269]: [82, 93], [298, 221, 48, 87]: [65, 22], [234, 237, 308]: [0], [298, 27, 112]: [79, 76, 39], [268, 187]: [65, 21, 165], [311]: [284, 202], [325, 34]: [12, 105], [317, 292]: [82, 152], [30]: [305, 271], [268, 237]: [287, 41], [268, 24]: [293, 8], [82, 7]: [79, 176], [82, 143]: [175, 52], [317, 108]: [65, 203], [298, 237, 79]: [65, 282, 303], [298, 77, 210]: [103], [298, 77, 64]: [27], [317, 345]: [55], [147, 138]: [284, 157], [249]: [65, 21, 207], [147, 127]: [175, 53], [337, 237, 41, 338]: [79, 76, 195], [240, 361]: [65, 272, 179], [354]: [79, 76, 307], [82, 67]: [306, 56], [215]: [65, 21], [298, 27, 362, 263]: [283, 238, 218, 278], [189, 71]: [156, 116], [298, 27]: [161], [102]: [215], [189, 259]: [163, 234], [298, 311]: [68, 15], [234, 237, 158, 110]: [90], [147, 138, 322]: [283, 238, 247, 277], [240, 248]: [161, 289], [165, 28]: [86], [234, 73]: [65, 21, 29], [234, 2]: [60], [207]: [68, 249], [298, 88, 175]: [283, 238, 218, 73], [189, 168]: [284], [298, 203]: [319, 4], [298, 77, 287]: [305, 199], [337, 156]: [68, 294], [258]: [3, 271], [317, 191]: [68, 16], [199, 152]: [293, 121], [132]: [68, 217], [337, 85]: [65, 21, 109], [268, 12]: [82, 162], [281]: [163, 118], [337, 337]: [302], [234, 304]: [283, 126], [30, 83]: [65, 297], [298]: [194], [147, 363, 291]: [283, 238, 218, 19], [174]: [68, 213, 120]}

    dict_lists_en_de = {}
    dict_lists_de_en = {}
    for element in dict_ids:
        dict_lists_en_de.update({dict_ids_en_rev[element]:dict_ids_de_rev[element]})
        dict_lists_de_en.update({dict_ids_de_rev[element]:dict_ids_en_rev[element]})

    print("dict_lists_en_de= ",dict_lists_en_de)
    print("dict_lists_de_en= ",dict_lists_de_en)

    with open("dict_lists_en_de.p","wb") as dict_pkl:
        pkl.dump(dict_lists_en_de,dict_pkl)

    with open("dict_lists_de_en.p","wb") as dict_pkl:
        pkl.dump(dict_lists_de_en,dict_pkl)

    return dict_lists_en_de, dict_lists_de_en

def generate_common_english_json():
    json_path_dev = "/aimlx/mhan/data/dw_general/dev/"
    json_path_train = "/aimlx/mhan/data/dw_general/train/"
    json_path_test = "/aimlx/mhan/data/dw_general/test/"

    en_train = process_json(json_path_train+"english.json")
    en_dev = process_json(json_path_dev+"english.json")
    en_test = process_json(json_path_test+"english.json")

    out_path = "/aimlx/mhan/common_data/dw_general/"

    # with open("dict_lists_en_de.p","rb") as dict_pkl:
    #     dict_lists_en_de = pkl.load(dict_pkl)
    #
    # with open("dict_lists_en_de.p","rb") as dict_pkl:
    #     dict_lists_en_de = pkl.load(dict_pkl)

    dict_lists_en_de, dict_lists_de_en = generate_dict_pkl()

    """ Train """
    indices_kept_train_en = []
    for i in range(0,len(en_train['Y_ids'])):
        y = en_train['Y_ids'][i]
        if tuple(y) in dict_lists_en_de:
            indices_kept_train_en.append(i)

    ## Dump new json file
    new_x_en_train = [en_train['X_ids'][i] for i in indices_kept_train_en]
    new_y_en_train = [en_train['Y_ids'][i] for i in indices_kept_train_en]

    train_en_file = open(out_path +"train/english.json", 'w')

    json.dump({'X_ids': new_x_en_train, 'Y_ids': new_y_en_train}, train_en_file)

    """ Dev """
    indices_kept_dev_en = []
    for i in range(0,len(en_dev['Y_ids'])):
        y = en_dev['Y_ids'][i]
        if tuple(y) in dict_lists_en_de:
            indices_kept_dev_en.append(i)

    new_x_en_dev = [en_dev['X_ids'][i] for i in indices_kept_dev_en]
    new_y_en_dev = [en_dev['Y_ids'][i]  for i in indices_kept_dev_en]

    dev_en_file = open(out_path +"dev/english.json", 'w')

    json.dump({'X_ids': new_x_en_dev, 'Y_ids': new_y_en_dev}, dev_en_file)

    """ Test """
    ## Dump new json file
    indices_kept_test_en = []
    for i in range(0,len(en_test['Y_ids'])):
        y = en_test['Y_ids'][i]
        if tuple(y) in dict_lists_en_de:
            indices_kept_test_en.append(i)

    new_x_en_test = [en_test['X_ids'][i] for i in indices_kept_test_en]
    new_y_en_test = [en_test['Y_ids'][i] for i in indices_kept_test_en]

    test_en_file = open(out_path +"test/english.json", 'w')

    json.dump({'X_ids': new_x_en_test, 'Y_ids': new_y_en_test}, test_en_file)

def generate_common_german_json ():
    json_path_dev = "/aimlx/mhan/data/dw_general/dev/"
    json_path_train = "/aimlx/mhan/data/dw_general/train/"
    json_path_test = "/aimlx/mhan/data/dw_general/test/"

    en_train = process_json(json_path_train+"german.json")
    en_dev = process_json(json_path_dev+"german.json")
    en_test = process_json(json_path_test+"german.json")

    out_path = "/aimlx/mhan/common_data/dw_general/"

    # with open("dict_lists_en_de.p","rb") as dict_pkl:
    #     dict_lists_en_de = pkl.load(dict_pkl)
    #
    # with open("dict_lists_en_de.p","rb") as dict_pkl:
    #     dict_lists_en_de = pkl.load(dict_pkl)

    dict_lists_en_de, dict_lists_de_en = generate_dict_pkl()

    """ Train """
    indices_kept_train_en = []
    for i in range(0,len(en_train['Y_ids'])):
        y = en_train['Y_ids'][i]
        if tuple(y) in dict_lists_de_en:
            indices_kept_train_en.append(i)

    ## Dump new json file
    new_x_en_train = [en_train['X_ids'][i] for i in indices_kept_train_en]
    new_y_en_train = [list(dict_lists_de_en[tuple(en_train['Y_ids'][i])]) for i in indices_kept_train_en]

    train_en_file = open(out_path +"train/german.json", 'w')

    json.dump({'X_ids': new_x_en_train, 'Y_ids': new_y_en_train}, train_en_file)

    """ Dev """
    indices_kept_dev_en = []
    for i in range(0,len(en_dev['Y_ids'])):
        y = en_dev['Y_ids'][i]
        if tuple(y) in dict_lists_de_en:
            indices_kept_dev_en.append(i)

    new_x_en_dev = [en_dev['X_ids'][i] for i in indices_kept_dev_en]
    new_y_en_dev = [list(dict_lists_de_en[tuple(en_dev['Y_ids'][i])])  for i in indices_kept_dev_en]

    dev_en_file = open(out_path +"dev/german.json", 'w')

    json.dump({'X_ids': new_x_en_dev, 'Y_ids': new_y_en_dev}, dev_en_file)

    """ Test """
    ## Dump new json file
    indices_kept_test_en = []
    for i in range(0,len(en_test['Y_ids'])):
        y = en_test['Y_ids'][i]
        if tuple(y) in dict_lists_de_en:
            indices_kept_test_en.append(i)

    new_x_en_test = [en_test['X_ids'][i] for i in indices_kept_test_en]
    new_y_en_test = [list(dict_lists_de_en[tuple(en_test['Y_ids'][i])])  for i in indices_kept_test_en]

    test_en_file = open(out_path +"test/german.json", 'w')

    json.dump({'X_ids': new_x_en_test, 'Y_ids': new_y_en_test}, test_en_file)

if __name__ == '__main__':
    # json_path_dev = "/aimlx/mhan/data/dw_general/dev/"
    # json_path_train = "/aimlx/mhan/data/dw_general/train/"
    # json_path_test = "/aimlx/mhan/data/dw_general/test/"
    #
    # word_vocab_path = "/aimlx/mhan/word_vectors/english.pkl"
    #
    # ## Read json files in English and German
    # en_train = process_json(json_path_train+"english.json")
    #
    # en_dev = process_json(json_path_dev+"english.json")
    #
    # en_test = process_json(json_path_test+"english.json")
    #
    # y_ids = en_train['Y_ids'] + en_dev['Y_ids'] +  en_test['Y_ids']
    #
    # y_ids_distinct_dev = list(set([item for sublist in en_dev['Y_ids'] for item in sublist]))
    #
    # y_ids_distinct = list(set([item for sublist in y_ids for item in sublist if item not in y_ids_distinct_dev]))
    #
    # print("len(en_train[label_ids]= ",len(en_train['label_ids']))
    # print("len(en_dev[label_ids]= ",len(en_dev['label_ids']))
    # print("len(en_test[label_ids]= ",len(en_test['label_ids']))
    #
    # print("y_ids_distinct= ",y_ids_distinct)
    # print("len(y_ids_distinct)= ",len(y_ids_distinct))
    #
    #
    # # print("y_ids_distinct= ",y_ids_distinct)
    # # print(len(en_dev['label_ids']))
    # #
    # with open(word_vocab_path) as en_vectors:
    #     emb_en = pkl.load(en_vectors)
    #
    # dict_en = {}
    # for i in range(0,len(emb_en[0])):
    #     dict_en.update({i:emb_en[0][i]})
    #
    # # print(dict_en)
    #
    # print("en_dev['label_ids']=",en_dev['label_ids'])
    #
    # labels_ids = set(en_train['label_ids'] + en_dev['label_ids'] +  en_test['label_ids'])
    #
    # print("labels_ids= ",labels_ids)
    #
    # print(len(labels_ids))
    # labels_distinct =  labels_ids - set(en_dev['label_ids'])
    #
    # print("label_ids_distinct_dev=",en_dev['label_ids'])
    # print("labels_distinct= ",labels_distinct)
    # labels_words = []
    # for label in labels_distinct:
    #     labels_words_sub = []
    #     print(label)
    #     for part in label.split("_"):
    #         labels_words_sub.append(dict_en[int(part)])
    #     labels_words.append(" ".join(labels_words_sub))
    #
    # print("labels_words= ",labels_words)
    #
    # print(labels_words[172].encode("utf-8"))
    # print(labels_words[283].encode("utf-8"))
    #
    # print(labels_words[55].encode("utf-8"))

    #print(labels_words[41].encode("utf-8"))
    #print(labels_words)

    # labels_en_de = []
    # ids_en_de = []
    # with open("/aimlx/MultiEmb_EventDet_Thesis/dataModule/dw_labels_en_translated_de.txt") as file:
    #     i = 0
    #     for line in file:
    #         if line[0] =='!':
    #             labels_en_de.append(line[2:-1])
    #             ids_en_de.append(i)
    #         if line[0]!='?' and line[0]!='!':
    #             labels_en_de.append(line[:-1])
    #             ids_en_de.append(i)
    #         i = i + 1
    #
    # labels_de = []
    # with open("/aimlx/MultiEmb_EventDet_Thesis/dataModule/dw_labels_german.txt") as file:
    #     for line in file:
    #         labels_de.append(line[:-1])
    #
    # ids_de = []
    # for label in labels_en_de:
    #     ids_de.append(labels_de.index(label))
    #
    #
    # dictionary_ids = dict(zip(ids_en_de, ids_de))
    #
    # print(dictionary_ids)
    #
    # labels_en = []
    # with open("/aimlx/MultiEmb_EventDet_Thesis/dataModule/dw_labels_english.txt") as file:
    #     for line in file:
    #         labels_en.append(line)
    #
    # labels_english = []
    # labels_german = []
    # for id in ids_en_de:
    #     labels_english.append(labels_en[id])
    #     labels_german.append(labels_de[dictionary_ids[id]])
    #
    #
    # dictionary_labels = dict(zip(labels_english, labels_german))
    #
    # print(dictionary_labels)

    json_path_dev = "/aimlx/mhan/data/dw_general/dev/"
    json_path_train = "/aimlx/mhan/data/dw_general/train/"
    json_path_test = "/aimlx/mhan/data/dw_general/test/"
    """ Mapping list of ids in English to English unique ids"""
    en_train = process_json(json_path_train+"german.json")
    en_dev = process_json(json_path_dev+"german.json")
    en_test = process_json(json_path_test+"german.json")

    out_path = "/aimlx/mhan/common_data/dw_general/"

    # with open("dict_lists_en_de.p","rb") as dict_pkl:
    #     dict_lists_en_de = pkl.load(dict_pkl)
    #
    # with open("dict_lists_en_de.p","rb") as dict_pkl:
    #     dict_lists_en_de = pkl.load(dict_pkl)

    dict_lists_en_de, dict_lists_de_en = generate_dict_pkl()

    """ Train """
    indices_kept_train_en = []
    for i in range(0,len(en_train['Y_ids'])):
        y = en_train['Y_ids'][i]
        if tuple(y) in dict_lists_de_en:
            indices_kept_train_en.append(i)

    ## Dump new json file
    new_x_en_train = [en_train['X_ids'][i] for i in indices_kept_train_en]
    new_y_en_train = [list(dict_lists_de_en[tuple(en_train['Y_ids'][i])]) for i in indices_kept_train_en]

    train_en_file = open(out_path +"train/german.json", 'w')

    json.dump({'X_ids': new_x_en_train, 'Y_ids': new_y_en_train}, train_en_file)

    """ Dev """
    indices_kept_dev_en = []
    for i in range(0,len(en_dev['Y_ids'])):
        y = en_dev['Y_ids'][i]
        if tuple(y) in dict_lists_de_en:
            indices_kept_dev_en.append(i)

    new_x_en_dev = [en_dev['X_ids'][i] for i in indices_kept_dev_en]
    new_y_en_dev = [list(dict_lists_de_en[tuple(en_dev['Y_ids'][i])])  for i in indices_kept_dev_en]

    dev_en_file = open(out_path +"dev/german.json", 'w')

    json.dump({'X_ids': new_x_en_dev, 'Y_ids': new_y_en_dev}, dev_en_file)

    """ Test """
    ## Dump new json file
    indices_kept_test_en = []
    for i in range(0,len(en_test['Y_ids'])):
        y = en_test['Y_ids'][i]
        if tuple(y) in dict_lists_de_en:
            indices_kept_test_en.append(i)

    new_x_en_test = [en_test['X_ids'][i] for i in indices_kept_test_en]
    new_y_en_test = [list(dict_lists_de_en[tuple(en_test['Y_ids'][i])])  for i in indices_kept_test_en]

    test_en_file = open(out_path +"test/german.json", 'w')

    json.dump({'X_ids': new_x_en_test, 'Y_ids': new_y_en_test}, test_en_file)









