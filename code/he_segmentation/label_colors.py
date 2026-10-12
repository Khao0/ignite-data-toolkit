"""RGB colors sampled from the supplied legend, in left-to-right order.

Legend order differs from model IDs: unannotated is ID 0, at the end.
"""
LEGEND = [
    (1, 'Background', (255, 255, 255)),
    (2, 'Tumor cells', (117, 248, 253)),
    (3, 'Reactive epithelium', (115, 245, 74)),
    (4, 'Stroma', (253, 250, 84)),
    (5, 'Inflammation', (239, 153, 56)),
    (6, 'Alveolar tissue', (234, 70, 144)),
    (7, 'Fatty tissue', (137, 62, 245)),
    (8, 'Necrotic tissue', (7, 24, 143)),
    (9, 'Erythrocytes', (232, 64, 38)),
    (10, 'Bronchial epithelium', (63, 141, 39)),
    (11, 'Mucus/Plasma/Fluids', (146, 143, 46)),
    (12, 'Cartilage/Bone', (169, 169, 169)),
    (13, 'Macrophages', (78, 31, 142)),
    (14, 'Muscle', (236, 133, 126)),
    (15, 'Liver', (202, 136, 245)),
    (16, 'Keratinization', (138, 36, 81)),
    (0, 'Unannotated', (0, 0, 0)),
]
COLORS = {class_id: rgb for class_id, _, rgb in LEGEND}
PALETTE = [COLORS[class_id] for class_id in range(17)]
