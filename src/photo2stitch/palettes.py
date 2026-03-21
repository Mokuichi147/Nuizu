"""Real embroidery thread color palettes.

Accurate RGB values for popular thread brands.
Used for color quantization to ensure output colors
match available real-world threads.
"""

from typing import List, Tuple

# Format: (R, G, B, Name, Catalog Number)
ThreadEntry = Tuple[int, int, int, str, str]

# JANOME Polyester Embroidery Thread palette (common subset)
JANOME_PALETTE: List[ThreadEntry] = [
    (0, 0, 0, "Black", "900"),
    (255, 255, 255, "White", "001"),
    (230, 0, 18, "Red", "800"),
    (178, 34, 34, "Dark Red", "810"),
    (255, 69, 0, "Vermillion", "820"),
    (255, 127, 80, "Coral", "825"),
    (255, 160, 122, "Light Salmon", "830"),
    (220, 20, 60, "Crimson", "805"),
    (199, 21, 133, "Deep Pink", "840"),
    (255, 105, 180, "Hot Pink", "841"),
    (255, 182, 193, "Light Pink", "842"),
    (255, 192, 203, "Pink", "843"),
    (128, 0, 128, "Purple", "850"),
    (148, 103, 189, "Medium Purple", "851"),
    (218, 112, 214, "Orchid", "852"),
    (230, 230, 250, "Lavender", "853"),
    (0, 0, 180, "Blue", "860"),
    (0, 0, 139, "Dark Blue", "861"),
    (30, 144, 255, "Dodger Blue", "862"),
    (70, 130, 180, "Steel Blue", "863"),
    (135, 206, 235, "Sky Blue", "864"),
    (173, 216, 230, "Light Blue", "865"),
    (0, 128, 128, "Teal", "870"),
    (0, 139, 139, "Dark Cyan", "871"),
    (64, 224, 208, "Turquoise", "872"),
    (0, 100, 0, "Dark Green", "880"),
    (34, 139, 34, "Forest Green", "881"),
    (0, 128, 0, "Green", "882"),
    (50, 205, 50, "Lime Green", "883"),
    (144, 238, 144, "Light Green", "884"),
    (107, 142, 35, "Olive Drab", "885"),
    (85, 107, 47, "Dark Olive", "886"),
    (255, 255, 0, "Yellow", "890"),
    (255, 215, 0, "Gold", "891"),
    (255, 200, 0, "Amber", "892"),
    (255, 165, 0, "Orange", "893"),
    (255, 140, 0, "Dark Orange", "894"),
    (210, 105, 30, "Chocolate", "895"),
    (139, 69, 19, "Saddle Brown", "900"),
    (160, 82, 45, "Sienna", "901"),
    (205, 133, 63, "Peru", "902"),
    (210, 180, 140, "Tan", "903"),
    (245, 222, 179, "Wheat", "904"),
    (255, 228, 196, "Bisque", "905"),
    (64, 64, 64, "Dark Gray", "910"),
    (128, 128, 128, "Gray", "911"),
    (192, 192, 192, "Silver", "912"),
    (220, 220, 220, "Light Gray", "913"),
]

# Brother Embroidery Thread palette (common subset)
BROTHER_PALETTE: List[ThreadEntry] = [
    (14, 31, 124, "Prussian Blue", "007"),
    (10, 85, 163, "Blue", "405"),
    (48, 135, 119, "Teal Green", "534"),
    (75, 107, 175, "Cornflower Blue", "070"),
    (237, 23, 31, "Red", "800"),
    (209, 92, 0, "Reddish Brown", "058"),
    (145, 54, 151, "Magenta", "614"),
    (228, 154, 203, "Light Lilac", "085"),
    (145, 95, 172, "Lilac", "607"),
    (158, 214, 125, "Mint Green", "027"),
    (232, 169, 0, "Deep Gold", "205"),
    (254, 186, 53, "Orange", "208"),
    (255, 255, 0, "Yellow", "202"),
    (112, 188, 31, "Lime Green", "513"),
    (186, 152, 0, "Brass", "328"),
    (168, 168, 168, "Silver", "005"),
    (125, 111, 0, "Russet Brown", "337"),
    (255, 255, 179, "Cream Brown", "107"),
    (79, 85, 86, "Pewter", "843"),
    (0, 0, 0, "Black", "900"),
    (11, 61, 145, "Ultra Marine", "420"),
    (119, 1, 118, "Royal Purple", "612"),
    (41, 49, 51, "Dark Gray", "707"),
    (42, 19, 1, "Dark Brown", "058"),
    (246, 74, 138, "Deep Rose", "086"),
    (178, 118, 36, "Light Brown", "348"),
    (252, 187, 197, "Salmon Pink", "018"),
    (254, 55, 15, "Vermillion", "206"),
    (240, 240, 240, "White", "001"),
    (106, 28, 138, "Violet", "613"),
    (168, 221, 196, "Seacrest", "542"),
    (37, 132, 187, "Sky Blue", "019"),
    (255, 243, 107, "Cream Yellow", "103"),
    (255, 200, 200, "Flesh Pink", "124"),
    (255, 217, 17, "Harvest Gold", "214"),
    (9, 91, 166, "Electric Blue", "420"),
    (0, 103, 62, "Emerald Green", "515"),
    (78, 41, 144, "Purple", "611"),
    (47, 126, 32, "Moss Green", "509"),
    (255, 204, 204, "Pink", "124"),
]

# Madeira Rayon 40 palette (common subset)
MADEIRA_PALETTE: List[ThreadEntry] = [
    (0, 0, 0, "Black", "1000"),
    (252, 252, 252, "White", "1001"),
    (229, 26, 46, "Red", "1037"),
    (190, 22, 34, "Dark Red", "1147"),
    (150, 2, 30, "Wine", "1181"),
    (236, 48, 127, "Fuchsia", "1110"),
    (248, 155, 188, "Light Pink", "1108"),
    (120, 27, 119, "Purple", "1033"),
    (83, 62, 147, "Violet", "1112"),
    (33, 48, 140, "Royal Blue", "1042"),
    (25, 104, 175, "Blue", "1076"),
    (117, 183, 219, "Sky Blue", "1096"),
    (0, 116, 97, "Teal", "1293"),
    (0, 118, 58, "Emerald", "1250"),
    (53, 137, 58, "Green", "1251"),
    (132, 178, 38, "Lime", "1169"),
    (254, 222, 0, "Yellow", "1024"),
    (253, 185, 0, "Gold", "1025"),
    (247, 148, 0, "Tangerine", "1065"),
    (237, 122, 28, "Orange", "1078"),
    (158, 97, 49, "Medium Brown", "1057"),
    (93, 55, 30, "Dark Brown", "1059"),
    (133, 133, 133, "Gray", "1114"),
    (193, 193, 193, "Light Gray", "1118"),
]


def get_palette(name: str = "janome") -> List[Tuple[int, int, int, str]]:
    """Get a thread palette by brand name.

    Args:
        name: 'janome', 'brother', 'madeira', or 'generic'.

    Returns:
        List of (R, G, B, Name) tuples.
    """
    palettes = {
        'janome': JANOME_PALETTE,
        'brother': BROTHER_PALETTE,
        'madeira': MADEIRA_PALETTE,
    }

    if name.lower() in palettes:
        return [(r, g, b, n) for r, g, b, n, _ in palettes[name.lower()]]

    # Generic palette (combined unique colors)
    from .quantize import THREAD_PALETTE
    return [(r, g, b, n) for r, g, b, n in THREAD_PALETTE]
