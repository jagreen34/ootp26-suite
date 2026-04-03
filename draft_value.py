"""
OOTP 26 Draft Pick Value Chart
================================
Pick-by-pick WAR projections, trade calculator, bust rates,
position analysis, and decision rules.

From 253K simulation dataset: 7,000 drafted players across
10 seeds × 5 draft classes × 28 teams.
"""

import numpy as np
import pandas as pd

# ============================================================================
# DATA (from 253K study)
# ============================================================================

PICK_DATA = [
{"pick":1,"round":1,"rp":1,"pts":3000,"raw":51.0,"smooth":43.4,"med":50.8,"pct25":92,"neg":2},{"pick":2,"round":1,"rp":2,"pts":2694,"raw":48.5,"smooth":39.0,"med":53.6,"pct25":66,"neg":0},{"pick":3,"round":1,"rp":3,"pts":2467,"raw":30.8,"smooth":35.7,"med":24.1,"pct25":48,"neg":2},{"pick":4,"round":1,"rp":4,"pts":2172,"raw":25.7,"smooth":31.4,"med":21.2,"pct25":40,"neg":0},{"pick":5,"round":1,"rp":5,"pts":1842,"raw":22.6,"smooth":26.7,"med":22.2,"pct25":42,"neg":0},
{"pick":6,"round":1,"rp":6,"pts":1739,"raw":29.6,"smooth":25.2,"med":26.0,"pct25":56,"neg":8},{"pick":7,"round":1,"rp":7,"pts":1832,"raw":24.7,"smooth":26.5,"med":21.1,"pct25":42,"neg":6},{"pick":8,"round":1,"rp":8,"pts":1837,"raw":23.4,"smooth":26.6,"med":20.3,"pct25":32,"neg":0},{"pick":9,"round":1,"rp":9,"pts":1709,"raw":32.4,"smooth":24.7,"med":21.5,"pct25":46,"neg":2},{"pick":10,"round":1,"rp":10,"pts":1609,"raw":23.0,"smooth":23.3,"med":14.7,"pct25":34,"neg":8},
{"pick":11,"round":1,"rp":11,"pts":1559,"raw":20.3,"smooth":22.6,"med":15.6,"pct25":32,"neg":6},{"pick":12,"round":1,"rp":12,"pts":1386,"raw":17.5,"smooth":20.1,"med":12.5,"pct25":22,"neg":8},{"pick":13,"round":1,"rp":13,"pts":1287,"raw":19.8,"smooth":18.6,"med":14.3,"pct25":26,"neg":6},{"pick":14,"round":1,"rp":14,"pts":1255,"raw":19.9,"smooth":18.2,"med":15.2,"pct25":32,"neg":6},{"pick":15,"round":1,"rp":15,"pts":1192,"raw":15.9,"smooth":17.3,"med":12.3,"pct25":22,"neg":8},
{"pick":16,"round":1,"rp":16,"pts":1151,"raw":17.9,"smooth":16.7,"med":14.8,"pct25":26,"neg":8},{"pick":17,"round":1,"rp":17,"pts":1142,"raw":12.9,"smooth":16.5,"med":10.6,"pct25":18,"neg":8},{"pick":18,"round":1,"rp":18,"pts":1145,"raw":16.8,"smooth":16.6,"med":14.5,"pct25":24,"neg":4},{"pick":19,"round":1,"rp":19,"pts":1097,"raw":19.2,"smooth":15.9,"med":14.2,"pct25":28,"neg":8},{"pick":20,"round":1,"rp":20,"pts":1100,"raw":16.0,"smooth":15.9,"med":11.1,"pct25":22,"neg":8},
{"pick":21,"round":1,"rp":21,"pts":1083,"raw":14.4,"smooth":15.7,"med":11.2,"pct25":20,"neg":8},{"pick":22,"round":1,"rp":22,"pts":1006,"raw":13.2,"smooth":14.6,"med":10.0,"pct25":18,"neg":10},{"pick":23,"round":1,"rp":23,"pts":979,"raw":15.6,"smooth":14.2,"med":11.2,"pct25":14,"neg":12},{"pick":24,"round":1,"rp":24,"pts":975,"raw":13.6,"smooth":14.1,"med":11.3,"pct25":18,"neg":8},{"pick":25,"round":1,"rp":25,"pts":1010,"raw":14.0,"smooth":14.6,"med":10.6,"pct25":18,"neg":8},
{"pick":26,"round":1,"rp":26,"pts":1008,"raw":14.2,"smooth":14.6,"med":9.9,"pct25":20,"neg":8},{"pick":27,"round":1,"rp":27,"pts":1000,"raw":15.7,"smooth":14.5,"med":11.7,"pct25":22,"neg":6},{"pick":28,"round":1,"rp":28,"pts":1058,"raw":15.4,"smooth":15.3,"med":12.6,"pct25":20,"neg":8},
{"pick":29,"round":2,"rp":1,"pts":1040,"raw":13.0,"smooth":15.1,"med":9.1,"pct25":16,"neg":8},{"pick":30,"round":2,"rp":2,"pts":974,"raw":18.2,"smooth":14.1,"med":12.5,"pct25":24,"neg":4},{"pick":31,"round":2,"rp":3,"pts":927,"raw":12.9,"smooth":13.4,"med":8.4,"pct25":14,"neg":8},{"pick":32,"round":2,"rp":4,"pts":902,"raw":10.9,"smooth":13.1,"med":7.7,"pct25":12,"neg":8},{"pick":33,"round":2,"rp":5,"pts":821,"raw":12.0,"smooth":11.9,"med":7.2,"pct25":12,"neg":6},{"pick":34,"round":2,"rp":6,"pts":826,"raw":11.2,"smooth":12.0,"med":8.0,"pct25":14,"neg":4},{"pick":35,"round":2,"rp":7,"pts":856,"raw":12.4,"smooth":12.4,"med":8.6,"pct25":16,"neg":4},{"pick":36,"round":2,"rp":8,"pts":893,"raw":13.2,"smooth":12.9,"med":9.6,"pct25":16,"neg":6},{"pick":37,"round":2,"rp":9,"pts":914,"raw":13.1,"smooth":13.2,"med":10.2,"pct25":18,"neg":4},{"pick":38,"round":2,"rp":10,"pts":918,"raw":14.6,"smooth":13.3,"med":10.7,"pct25":18,"neg":4},
{"pick":39,"round":2,"rp":11,"pts":913,"raw":12.7,"smooth":13.2,"med":9.3,"pct25":14,"neg":6},{"pick":40,"round":2,"rp":12,"pts":992,"raw":12.8,"smooth":14.4,"med":9.1,"pct25":14,"neg":6},{"pick":41,"round":2,"rp":13,"pts":1029,"raw":12.9,"smooth":14.9,"med":8.2,"pct25":14,"neg":8},{"pick":42,"round":2,"rp":14,"pts":1018,"raw":18.8,"smooth":14.7,"med":12.7,"pct25":36,"neg":0},{"pick":43,"round":2,"rp":15,"pts":1059,"raw":17.4,"smooth":15.3,"med":12.1,"pct25":20,"neg":2},{"pick":44,"round":2,"rp":16,"pts":1053,"raw":11.9,"smooth":15.2,"med":10.1,"pct25":16,"neg":2},{"pick":45,"round":2,"rp":17,"pts":1016,"raw":15.7,"smooth":14.7,"med":10.3,"pct25":18,"neg":4},{"pick":46,"round":2,"rp":18,"pts":930,"raw":12.5,"smooth":13.5,"med":9.0,"pct25":14,"neg":6},{"pick":47,"round":2,"rp":19,"pts":975,"raw":16.2,"smooth":14.1,"med":12.0,"pct25":20,"neg":2},{"pick":48,"round":2,"rp":20,"pts":931,"raw":11.1,"smooth":13.5,"med":8.4,"pct25":14,"neg":4},
{"pick":49,"round":2,"rp":21,"pts":982,"raw":15.1,"smooth":14.2,"med":9.4,"pct25":16,"neg":4},{"pick":50,"round":2,"rp":22,"pts":976,"raw":12.6,"smooth":14.1,"med":9.7,"pct25":14,"neg":6},{"pick":51,"round":2,"rp":23,"pts":1020,"raw":16.2,"smooth":14.8,"med":11.0,"pct25":16,"neg":2},{"pick":52,"round":2,"rp":24,"pts":996,"raw":15.7,"smooth":14.4,"med":11.1,"pct25":18,"neg":4},{"pick":53,"round":2,"rp":25,"pts":983,"raw":14.3,"smooth":14.2,"med":11.2,"pct25":16,"neg":4},{"pick":54,"round":2,"rp":26,"pts":940,"raw":13.4,"smooth":13.6,"med":9.9,"pct25":14,"neg":4},{"pick":55,"round":2,"rp":27,"pts":850,"raw":11.7,"smooth":12.3,"med":7.5,"pct25":12,"neg":6},{"pick":56,"round":2,"rp":28,"pts":826,"raw":13.0,"smooth":12.0,"med":7.5,"pct25":18,"neg":6},
{"pick":57,"round":3,"rp":1,"pts":789,"raw":9.2,"smooth":11.4,"med":5.9,"pct25":6,"neg":6},{"pick":58,"round":3,"rp":2,"pts":811,"raw":12.5,"smooth":11.7,"med":8.7,"pct25":12,"neg":4},{"pick":59,"round":3,"rp":3,"pts":790,"raw":10.7,"smooth":11.4,"med":8.2,"pct25":10,"neg":6},{"pick":60,"round":3,"rp":4,"pts":811,"raw":13.3,"smooth":11.7,"med":9.5,"pct25":12,"neg":4},{"pick":61,"round":3,"rp":5,"pts":784,"raw":11.5,"smooth":11.3,"med":8.4,"pct25":10,"neg":6},{"pick":62,"round":3,"rp":6,"pts":783,"raw":10.7,"smooth":11.3,"med":7.6,"pct25":10,"neg":6},{"pick":63,"round":3,"rp":7,"pts":755,"raw":10.6,"smooth":10.9,"med":7.4,"pct25":10,"neg":6},{"pick":64,"round":3,"rp":8,"pts":751,"raw":10.6,"smooth":10.9,"med":7.6,"pct25":8,"neg":6},{"pick":65,"round":3,"rp":9,"pts":721,"raw":11.2,"smooth":10.4,"med":7.8,"pct25":8,"neg":4},{"pick":66,"round":3,"rp":10,"pts":716,"raw":11.2,"smooth":10.4,"med":7.8,"pct25":10,"neg":4},
{"pick":67,"round":3,"rp":11,"pts":702,"raw":8.5,"smooth":10.2,"med":5.9,"pct25":8,"neg":6},{"pick":68,"round":3,"rp":12,"pts":705,"raw":10.2,"smooth":10.2,"med":6.8,"pct25":8,"neg":6},{"pick":69,"round":3,"rp":13,"pts":696,"raw":9.6,"smooth":10.1,"med":6.8,"pct25":8,"neg":6},{"pick":70,"round":3,"rp":14,"pts":784,"raw":11.5,"smooth":11.4,"med":8.2,"pct25":14,"neg":10},{"pick":71,"round":3,"rp":15,"pts":787,"raw":10.6,"smooth":11.4,"med":8.0,"pct25":12,"neg":8},{"pick":72,"round":3,"rp":16,"pts":776,"raw":14.9,"smooth":11.2,"med":10.2,"pct25":14,"neg":2},{"pick":73,"round":3,"rp":17,"pts":749,"raw":10.4,"smooth":10.9,"med":7.7,"pct25":10,"neg":6},{"pick":74,"round":3,"rp":18,"pts":771,"raw":8.9,"smooth":11.2,"med":6.3,"pct25":10,"neg":8},{"pick":75,"round":3,"rp":19,"pts":737,"raw":9.5,"smooth":10.7,"med":7.1,"pct25":8,"neg":8},{"pick":76,"round":3,"rp":20,"pts":696,"raw":12.1,"smooth":10.1,"med":9.4,"pct25":12,"neg":6},
{"pick":77,"round":3,"rp":21,"pts":711,"raw":12.4,"smooth":10.3,"med":8.3,"pct25":10,"neg":4},{"pick":78,"round":3,"rp":22,"pts":703,"raw":7.4,"smooth":10.2,"med":6.0,"pct25":8,"neg":8},{"pick":79,"round":3,"rp":23,"pts":651,"raw":9.9,"smooth":9.4,"med":7.4,"pct25":6,"neg":8},{"pick":80,"round":3,"rp":24,"pts":599,"raw":8.9,"smooth":8.7,"med":6.0,"pct25":4,"neg":8},{"pick":81,"round":3,"rp":25,"pts":673,"raw":8.4,"smooth":9.7,"med":6.2,"pct25":4,"neg":8},{"pick":82,"round":3,"rp":26,"pts":642,"raw":8.6,"smooth":9.3,"med":6.1,"pct25":4,"neg":8},{"pick":83,"round":3,"rp":27,"pts":614,"raw":12.8,"smooth":8.9,"med":9.0,"pct25":8,"neg":6},{"pick":84,"round":3,"rp":28,"pts":567,"raw":7.8,"smooth":8.2,"med":7.5,"pct25":4,"neg":8},
{"pick":85,"round":4,"rp":1,"pts":544,"raw":6.8,"smooth":7.9,"med":4.4,"pct25":6,"neg":4},{"pick":86,"round":4,"rp":2,"pts":450,"raw":5.0,"smooth":6.5,"med":3.1,"pct25":4,"neg":4},{"pick":87,"round":4,"rp":3,"pts":415,"raw":6.9,"smooth":6.0,"med":4.8,"pct25":4,"neg":4},{"pick":88,"round":4,"rp":4,"pts":437,"raw":6.0,"smooth":6.3,"med":4.4,"pct25":4,"neg":4},{"pick":89,"round":4,"rp":5,"pts":443,"raw":5.2,"smooth":6.4,"med":3.7,"pct25":4,"neg":4},{"pick":90,"round":4,"rp":6,"pts":456,"raw":8.4,"smooth":6.6,"med":5.8,"pct25":6,"neg":4},{"pick":91,"round":4,"rp":7,"pts":450,"raw":5.5,"smooth":6.5,"med":4.1,"pct25":4,"neg":4},{"pick":92,"round":4,"rp":8,"pts":472,"raw":7.9,"smooth":6.8,"med":5.2,"pct25":6,"neg":4},{"pick":93,"round":4,"rp":9,"pts":433,"raw":5.6,"smooth":6.3,"med":4.0,"pct25":4,"neg":4},{"pick":94,"round":4,"rp":10,"pts":435,"raw":6.8,"smooth":6.3,"med":5.0,"pct25":4,"neg":4},
{"pick":95,"round":4,"rp":11,"pts":420,"raw":5.6,"smooth":6.1,"med":3.8,"pct25":2,"neg":4},{"pick":96,"round":4,"rp":12,"pts":418,"raw":5.7,"smooth":6.1,"med":4.3,"pct25":2,"neg":4},{"pick":97,"round":4,"rp":13,"pts":396,"raw":6.8,"smooth":5.7,"med":4.7,"pct25":2,"neg":6},{"pick":98,"round":4,"rp":14,"pts":393,"raw":5.5,"smooth":5.7,"med":3.6,"pct25":2,"neg":6},{"pick":99,"round":4,"rp":15,"pts":378,"raw":5.2,"smooth":5.5,"med":3.7,"pct25":2,"neg":4},{"pick":100,"round":4,"rp":16,"pts":344,"raw":5.4,"smooth":5.0,"med":3.7,"pct25":2,"neg":4},{"pick":101,"round":4,"rp":17,"pts":334,"raw":4.6,"smooth":4.8,"med":3.3,"pct25":2,"neg":4},{"pick":102,"round":4,"rp":18,"pts":349,"raw":4.3,"smooth":5.1,"med":3.1,"pct25":2,"neg":4},{"pick":103,"round":4,"rp":19,"pts":352,"raw":4.7,"smooth":5.1,"med":3.4,"pct25":2,"neg":4},{"pick":104,"round":4,"rp":20,"pts":360,"raw":6.2,"smooth":5.2,"med":4.3,"pct25":2,"neg":4},
{"pick":105,"round":4,"rp":21,"pts":371,"raw":5.6,"smooth":5.4,"med":4.0,"pct25":2,"neg":4},{"pick":106,"round":4,"rp":22,"pts":393,"raw":5.2,"smooth":5.7,"med":3.5,"pct25":2,"neg":4},{"pick":107,"round":4,"rp":23,"pts":382,"raw":5.1,"smooth":5.5,"med":3.5,"pct25":2,"neg":4},{"pick":108,"round":4,"rp":24,"pts":381,"raw":6.3,"smooth":5.5,"med":4.5,"pct25":2,"neg":4},{"pick":109,"round":4,"rp":25,"pts":397,"raw":5.5,"smooth":5.7,"med":3.8,"pct25":2,"neg":4},{"pick":110,"round":4,"rp":26,"pts":399,"raw":5.5,"smooth":5.8,"med":3.8,"pct25":2,"neg":4},{"pick":111,"round":4,"rp":27,"pts":390,"raw":6.4,"smooth":5.6,"med":4.3,"pct25":2,"neg":4},{"pick":112,"round":4,"rp":28,"pts":379,"raw":5.2,"smooth":5.5,"med":3.5,"pct25":2,"neg":4},
{"pick":113,"round":5,"rp":1,"pts":362,"raw":5.7,"smooth":5.2,"med":3.0,"pct25":2,"neg":8},{"pick":114,"round":5,"rp":2,"pts":367,"raw":4.7,"smooth":5.3,"med":2.7,"pct25":2,"neg":8},{"pick":115,"round":5,"rp":3,"pts":375,"raw":4.2,"smooth":5.4,"med":2.5,"pct25":2,"neg":8},{"pick":116,"round":5,"rp":4,"pts":375,"raw":6.7,"smooth":5.4,"med":4.1,"pct25":4,"neg":6},{"pick":117,"round":5,"rp":5,"pts":387,"raw":5.8,"smooth":5.6,"med":3.9,"pct25":4,"neg":6},{"pick":118,"round":5,"rp":6,"pts":421,"raw":5.7,"smooth":6.1,"med":3.9,"pct25":4,"neg":4},{"pick":119,"round":5,"rp":7,"pts":408,"raw":5.5,"smooth":5.9,"med":3.8,"pct25":4,"neg":4},{"pick":120,"round":5,"rp":8,"pts":420,"raw":6.8,"smooth":6.1,"med":4.6,"pct25":4,"neg":4},{"pick":121,"round":5,"rp":9,"pts":428,"raw":5.7,"smooth":6.2,"med":3.8,"pct25":4,"neg":4},{"pick":122,"round":5,"rp":10,"pts":414,"raw":6.7,"smooth":6.0,"med":4.6,"pct25":4,"neg":4},
{"pick":123,"round":5,"rp":11,"pts":404,"raw":6.3,"smooth":5.9,"med":4.2,"pct25":2,"neg":4},{"pick":124,"round":5,"rp":12,"pts":400,"raw":4.5,"smooth":5.8,"med":3.2,"pct25":2,"neg":6},{"pick":125,"round":5,"rp":13,"pts":412,"raw":6.1,"smooth":6.0,"med":4.1,"pct25":2,"neg":4},{"pick":126,"round":5,"rp":14,"pts":464,"raw":5.4,"smooth":6.7,"med":3.6,"pct25":2,"neg":4},{"pick":127,"round":5,"rp":15,"pts":488,"raw":7.5,"smooth":7.1,"med":5.1,"pct25":4,"neg":4},{"pick":128,"round":5,"rp":16,"pts":501,"raw":10.1,"smooth":7.3,"med":7.2,"pct25":4,"neg":4},{"pick":129,"round":5,"rp":17,"pts":521,"raw":6.2,"smooth":7.5,"med":4.6,"pct25":4,"neg":4},{"pick":130,"round":5,"rp":18,"pts":486,"raw":7.0,"smooth":7.0,"med":5.1,"pct25":4,"neg":4},{"pick":131,"round":5,"rp":19,"pts":446,"raw":6.9,"smooth":6.5,"med":5.0,"pct25":4,"neg":4},{"pick":132,"round":5,"rp":20,"pts":434,"raw":5.0,"smooth":6.3,"med":3.4,"pct25":2,"neg":4},
{"pick":133,"round":5,"rp":21,"pts":445,"raw":7.1,"smooth":6.4,"med":4.8,"pct25":2,"neg":4},{"pick":134,"round":5,"rp":22,"pts":436,"raw":5.4,"smooth":6.3,"med":3.7,"pct25":2,"neg":4},{"pick":135,"round":5,"rp":23,"pts":461,"raw":7.8,"smooth":6.7,"med":5.6,"pct25":2,"neg":4},{"pick":136,"round":5,"rp":24,"pts":447,"raw":6.2,"smooth":6.5,"med":4.2,"pct25":2,"neg":4},{"pick":137,"round":5,"rp":25,"pts":461,"raw":6.9,"smooth":6.7,"med":4.8,"pct25":2,"neg":4},{"pick":138,"round":5,"rp":26,"pts":455,"raw":6.1,"smooth":6.6,"med":4.2,"pct25":2,"neg":4},{"pick":139,"round":5,"rp":27,"pts":462,"raw":6.4,"smooth":6.7,"med":4.4,"pct25":2,"neg":4},{"pick":140,"round":5,"rp":28,"pts":457,"raw":7.4,"smooth":6.6,"med":5.1,"pct25":2,"neg":0},
]

ROUND_DATA = [
{"round":"Round 1","n":1400,"avg":21.6,"med":15.5,"p75":31.8,"p25":6.2,"pct10":62.8,"pct20":40.3,"neg":6.4,"under5":22.4,"five15":26.5,"fifteen25":18.9,"star25":32.2},
{"round":"Round 2","n":1400,"avg":13.8,"med":9.5,"p75":18.4,"p25":2.8,"pct10":49.1,"pct20":22.9,"neg":3.8,"under5":34.7,"five15":30.8,"fifteen25":16.9,"star25":17.6},
{"round":"Round 3","n":1400,"avg":10.5,"med":7.4,"p75":14.2,"p25":1.9,"pct10":39.9,"pct20":13.6,"neg":5.4,"under5":39.6,"five15":37.5,"fifteen25":12.6,"star25":10.2},
{"round":"Round 4","n":1400,"avg":5.8,"med":3.7,"p75":8.6,"p25":0.0,"pct10":21.7,"pct20":3.6,"neg":3.9,"under5":56.9,"five15":35.2,"fifteen25":5.4,"star25":2.5},
{"round":"Round 5","n":1400,"avg":6.3,"med":3.6,"p75":9.5,"p25":0.0,"pct10":23.3,"pct20":5.4,"neg":4.7,"under5":59.1,"five15":32.4,"fifteen25":4.9,"star25":3.7},
]

BUCKET_DATA = [
{"bucket":"1-5","n":250,"avg":35.7,"med":31.4,"neg":0.8,"under5":9.2,"five15":18.0,"fifteen25":15.2,"star25":57.6},
{"bucket":"6-10","n":250,"avg":26.6,"med":20.8,"neg":4.8,"under5":12.8,"five15":26.0,"fifteen25":19.2,"star25":42.0},
{"bucket":"11-15","n":250,"avg":18.6,"med":13.4,"neg":7.6,"under5":26.0,"five15":26.0,"fifteen25":20.8,"star25":27.2},
{"bucket":"16-20","n":250,"avg":16.6,"med":12.1,"neg":9.2,"under5":27.6,"five15":28.8,"fifteen25":21.2,"star25":22.4},
{"bucket":"21-28","n":400,"avg":14.5,"med":10.9,"neg":8.3,"under5":31.0,"five15":31.0,"fifteen25":18.5,"star25":19.5},
{"bucket":"R2 (29-56)","n":1400,"avg":13.8,"med":9.5,"neg":3.8,"under5":34.7,"five15":30.8,"fifteen25":16.9,"star25":17.6},
{"bucket":"R3 (57-84)","n":1400,"avg":10.5,"med":7.4,"neg":5.4,"under5":39.6,"five15":37.5,"fifteen25":12.6,"star25":10.2},
{"bucket":"R4 (85-112)","n":1400,"avg":5.8,"med":3.7,"neg":3.9,"under5":56.9,"five15":35.2,"fifteen25":5.4,"star25":2.5},
{"bucket":"R5 (113-140)","n":1400,"avg":6.3,"med":3.6,"neg":4.7,"under5":59.1,"five15":32.4,"fifteen25":4.9,"star25":3.7},
]

POS_DATA = [
{"pos":"SS","avg":23.4,"med":17.5,"n":481,"tier":"Premium"},
{"pos":"CF","avg":18.3,"med":12.7,"n":733,"tier":"Premium"},
{"pos":"RF","avg":13.9,"med":10.1,"n":291,"tier":"Solid"},
{"pos":"3B","avg":13.1,"med":7.8,"n":354,"tier":"Solid"},
{"pos":"2B","avg":12.1,"med":7.7,"n":367,"tier":"Solid"},
{"pos":"LF","avg":11.0,"med":5.2,"n":252,"tier":"Average"},
{"pos":"SP","avg":10.7,"med":6.8,"n":2355,"tier":"Average"},
{"pos":"C","avg":9.1,"med":6.4,"n":698,"tier":"Average"},
{"pos":"1B","avg":7.4,"med":4.2,"n":297,"tier":"Below"},
{"pos":"RP","avg":5.7,"med":3.0,"n":1172,"tier":"Below"},
]

EXCHANGE_RATES = [
{"label": "Pick #1 = how many R2 picks?", "val": "2.3x"},
{"label": "Pick #1 = how many R3 picks?", "val": "3.0x"},
{"label": "1st round avg = R2 avg?", "val": "1.6x"},
{"label": "1st round avg = R3 avg?", "val": "2.1x"},
{"label": "Top-10 pick = R2 picks?", "val": "2.3x"},
{"label": "Top-10 pick = R3 picks?", "val": "3.0x"},
{"label": "Pick #28 vs #29 premium?", "val": "1.2x"},
{"label": "Round 2-3 boundary cliff?", "val": "1.4x"},
]


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

_PICK_LOOKUP = {p["pick"]: p for p in PICK_DATA}

def get_pick(pick_num):
    return _PICK_LOOKUP.get(pick_num)

def pts_for_pick(pick_num):
    p = get_pick(pick_num)
    return p["pts"] if p else 50

def evaluate_trade(side_a_picks, side_b_picks):
    """Evaluate a pick-for-pick trade. Returns dict with totals and verdict."""
    pts_a = sum(pts_for_pick(p) for p in side_a_picks)
    pts_b = sum(pts_for_pick(p) for p in side_b_picks)
    war_a = sum((get_pick(p) or {"raw":0})["raw"] for p in side_a_picks)
    war_b = sum((get_pick(p) or {"raw":0})["raw"] for p in side_b_picks)
    diff = pts_a - pts_b
    if abs(diff) < max(pts_a, pts_b, 1) * 0.08:
        verdict = "FAIR"
    elif diff > 0:
        verdict = "TEAM A OVERPAYS"
    else:
        verdict = "TEAM B OVERPAYS"
    return {"pts_a": pts_a, "pts_b": pts_b, "war_a": war_a, "war_b": war_b,
            "diff": diff, "verdict": verdict}


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_draft_value():
    try:
        import streamlit as st
    except ImportError:
        return

    st.header("📊 Draft Pick Value Chart")
    st.caption("253K dataset: 7,000 drafted players across 10 seeds x 5 draft classes x 28 teams")

    tab_curve, tab_trade, tab_rounds, tab_bust, tab_pos, tab_rules = st.tabs([
        "Value Curve", "Trade Calculator", "Round Summary", "Bust Rates", "By Position", "Decision Rules"
    ])

    # ==================== VALUE CURVE ====================
    with tab_curve:
        pick_df = pd.DataFrame(PICK_DATA)
        st.subheader("Career WAR by Overall Pick")
        st.line_chart(pick_df.set_index("pick")[["smooth", "raw"]], use_container_width=True)

        st.subheader("Pick Value Tiers")
        tier_data = [
            {"Tier": "PREMIUM", "Picks": "1-10", "Avg WAR": "27-51", "Star Rate": "42-58%", "Bust Rate": "0-8%"},
            {"Tier": "HIGH VALUE", "Picks": "11-28", "Avg WAR": "13-20", "Star Rate": "19-27%", "Bust Rate": "6-12%"},
            {"Tier": "LOTTERY", "Picks": "29-84 (R2-R3)", "Avg WAR": "10-14", "Star Rate": "10-18%", "Bust Rate": "4-8%"},
            {"Tier": "VOLUME", "Picks": "85-140 (R4-R5)", "Avg WAR": "5-6", "Star Rate": "2-4%", "Bust Rate": "4-8%"},
        ]
        st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

        st.subheader("Points Lookup")
        selected_round = st.selectbox("Round", [1, 2, 3, 4, 5], key="pts_round")
        round_picks = [p for p in PICK_DATA if p["round"] == selected_round]
        pts_df = pd.DataFrame(round_picks)[["pick", "rp", "pts", "raw", "smooth", "med", "pct25", "neg"]]
        pts_df.columns = ["OA Pick", "Round Pick", "Points", "Avg WAR", "Smooth WAR", "Median", "Star 25+%", "Bust%"]
        st.dataframe(pts_df, use_container_width=True, hide_index=True)

    # ==================== TRADE CALCULATOR ====================
    with tab_trade:
        st.subheader("Pick Trade Calculator")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Team A Gives**")
            a_input = st.text_input("Team A picks (comma-separated)", "1", key="trade_a")
        with col2:
            st.markdown("**Team B Gives**")
            b_input = st.text_input("Team B picks (comma-separated)", "29, 57", key="trade_b")

        try:
            a_picks = [int(x.strip()) for x in a_input.split(",") if x.strip()]
            b_picks = [int(x.strip()) for x in b_input.split(",") if x.strip()]
        except ValueError:
            a_picks, b_picks = [], []

        if a_picks and b_picks:
            result = evaluate_trade(a_picks, b_picks)

            vc = "🟢" if result["verdict"] == "FAIR" else "🔴" if "A" in result["verdict"] else "🟠"
            st.markdown(f"### {vc} {result['verdict']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Team A Total", f"{result['pts_a']:,} pts / {result['war_a']:.1f} WAR")
            c2.metric("Team B Total", f"{result['pts_b']:,} pts / {result['war_b']:.1f} WAR")
            c3.metric("Difference", f"{abs(result['diff']):,} pts / {abs(result['war_a']-result['war_b']):.1f} WAR")

            # Show pick details
            for label, picks in [("Team A", a_picks), ("Team B", b_picks)]:
                rows = []
                for p in picks:
                    d = get_pick(p)
                    if d:
                        rows.append({"Pick": f"#{p}", "Round": f"R{d['round']}P{d['rp']}", "Points": d["pts"],
                                    "Avg WAR": d["raw"], "Star%": d["pct25"]})
                if rows:
                    st.markdown(f"**{label} picks:**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Exchange Rates")
        st.dataframe(pd.DataFrame(EXCHANGE_RATES), use_container_width=True, hide_index=True)

    # ==================== ROUND SUMMARY ====================
    with tab_rounds:
        st.subheader("Performance by Round")
        rd = pd.DataFrame(ROUND_DATA)
        display_rd = rd[["round", "n", "avg", "med", "p75", "p25", "star25", "fifteen25", "five15", "under5", "neg"]]
        display_rd.columns = ["Round", "N", "Avg WAR", "Median", "Top 25%", "Bot 25%", "Star 25+%", "Starter 15-25%", "Contrib 5-15%", "Disappoint <5%", "Bust <0%"]
        st.dataframe(display_rd, use_container_width=True, hide_index=True)

        st.subheader("Performance by Pick Range")
        bd = pd.DataFrame(BUCKET_DATA)
        display_bd = bd[["bucket", "n", "avg", "med", "star25", "fifteen25", "five15", "under5", "neg"]]
        display_bd.columns = ["Pick Range", "N", "Avg WAR", "Median", "Star 25+%", "Starter%", "Contrib%", "Disappoint%", "Bust%"]
        st.dataframe(display_bd, use_container_width=True, hide_index=True)

    # ==================== BUST RATES ====================
    with tab_bust:
        st.subheader("Outcome Distribution by Pick Range")
        bd = pd.DataFrame(BUCKET_DATA)
        st.bar_chart(bd.set_index("bucket")[["star25", "fifteen25", "five15", "under5", "neg"]], use_container_width=True)

        st.markdown("**Key findings:**")
        st.markdown(
            "- **Picks 1-5:** 57.6% stars, 0.8% bust — most reliable assets in the game\n"
            "- **Picks 6-28:** Bust rate 4.8-9.2%, star rate 19-42% — scouting matters most here\n"
            "- **Round 2:** Lower bust rate (3.8%) than picks 11-28 (7.6-9.2%) — better floor\n"
            "- **R3→R4 is the steepest cliff:** R3 avg 10.5 WAR vs R4 avg 5.8 WAR (-45%)\n"
            "- **Rounds 4-5:** 57-59% produce <5 career WAR — draft for upside, expect 1 contributor per 4 picks"
        )

    # ==================== BY POSITION ====================
    with tab_pos:
        st.subheader("Career WAR by Draft Position")
        pos_df = pd.DataFrame(POS_DATA)
        display_pos = pos_df[["pos", "tier", "avg", "med", "n"]]
        display_pos.columns = ["Position", "Tier", "Avg Career WAR", "Median WAR", "N Drafted"]
        st.dataframe(display_pos, use_container_width=True, hide_index=True)

        st.markdown("**Key findings:**")
        st.markdown(
            "- **SS premium is massive:** 23.4 avg WAR — 4.1x more than RP (5.7)\n"
            "- **CF second-best** at 18.3 avg WAR\n"
            "- **RF > LF** (13.9 vs 11.0) — right field undervalued in drafts\n"
            "- **RP and 1B are pick-wasters** — draft only in R4-R5 with exceptional F1\n"
            "- **SP is most common** (n=2,355) but only avg 10.7 WAR — nearly same as LF"
        )

        st.subheader("Batter vs Pitcher by Round")
        bvp = pd.DataFrame([
            {"Round": "Round 1", "Batter": 22.5, "Pitcher": 19.9},
            {"Round": "Round 2", "Batter": 15.4, "Pitcher": 11.0},
            {"Round": "Round 3", "Batter": 11.4, "Pitcher": 9.6},
            {"Round": "Round 4", "Batter": 8.4, "Pitcher": 5.6},
            {"Round": "Round 5", "Batter": 7.1, "Pitcher": 5.0},
        ])
        st.dataframe(bvp, use_container_width=True, hide_index=True)
        st.markdown("**Exception:** Picks 6-10 favor pitchers (30.3 vs 25.5 WAR). Everywhere else, batters win.")

    # ==================== DECISION RULES ====================
    with tab_rules:
        st.subheader("Trade Decision Rules")
        rules = [
            ("Trade down at picks 22-28", "Value curve flattens. Two 2nd rounders (27.6 WAR) > one late 1st (13-15 WAR)."),
            ("Protect top-5 picks aggressively", "35.7 avg WAR, 57.6% star rate. Need 2.6x R2 picks or 3.4x R3 picks to match. Irreplaceable."),
            ("3.0x R3 = 1 top-10 pick", "Top-10 avg 31.2 WAR. R3 avg 10.5 WAR. Rarely have the volume to compensate."),
            ("Real cliff is at pick #10, not #28", "Pick #28 (15.4 WAR) vs #29 (13.0) = only 1.2x. Pick #10 (23.0) vs #11 (20.3) = steeper drop."),
            ("R3→R4 is steepest cliff", "R3 avg 10.5 vs R4 avg 5.8 = 45% collapse. R3 has star potential; R4 is coin flips."),
            ("In hoarding meta: overpay 20% for top-10", "When GMs won't trade players under 28, picks are the only path to pre-peak talent."),
        ]
        for title, detail in rules:
            st.markdown(f"**{title}**")
            st.markdown(f"{detail}")
            st.markdown("---")
