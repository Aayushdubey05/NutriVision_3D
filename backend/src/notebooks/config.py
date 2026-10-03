# Predefined Indian tableware sizes (diameter in mm)
TABLEWARE_PRESETS = {
    "standard_thali": 280.0,
    "grand_thali": 330.0,
    "quarter_plate": 200.0,
    "katori_bowl": 95.0,
}

PLATE_DEFAULTS = {
    "standard_thali": 280.0,
    "grand_thali": 330.0,
    "quarter_plate": 200.0,
    "katori_bowl": 95.0,
}

PLATE_DETECTION = {
    "min_circularity": 0.2,
    "min_area_fraction": 0.05,
    "max_area_fraction": 1.0,
    "plate_thickness_mm": 3.0,
    "assumed_plate_food_height_mm": 25.0,
    "min_confidence_threshold": 0.3,
    "scoring_weights": {
        "area": 0.3,
        "circularity": 0.1,
        "centered": 0.1,
        "border_touch": 0.5,
    }
}

CLASSIFICATION_CONFIG = {
    "confidence_threshold": 0.35,
    "top_k": 3,
    "class_merge_map": {
        "aloo_group": [
            "aloo_gobi",
            "aloo_matar",
            "aloo_methi",
            "aloo_shimla_mirch",
            "aloo_tikki",
            "dum_aloo",
        ],
        "chicken_group": [
            "butter_chicken",
            "chicken_tikka",
            "chicken_tikka_masala",
            "chicken_razala",
        ],
        "sweet_white_group": [
            "rasgulla",
            "cham_cham",
            "ras_malai",
            "sandesh",
            "chhena_kheeri",
        ],
        "sweet_brown_group": [
            "gulab_jamun",
            "jalebi",
            "imarti",
            "malapua",
            "double_ka_meetha",
        ],
        "sweet_fried_group": [
            "adhirasam",
            "ariselu",
            "anarsa",
            "bandar_laddu",
            "unni_appam",
            "kajjikaya",
            "kakinada_khaja",
            "ghevar",
            "gavvalu",
        ],
        "bread_group": [
            "roti",
            "chapati",
            "naan",
            "bhatura",
            "misi_roti",
            "makki_di_roti_sarson_da_saag",
        ],
        "dal_group": [
            "dal_makhani",
            "dal_tadka",
            "chana_masala",
        ],
        "rice_group": [
            "biryani",
            "daal_baati_churma",
        ],
    },
    "merge_targets": {
        "aloo_gobi": "aloo_group",
        "aloo_matar": "aloo_group",
        "aloo_methi": "aloo_group",
        "aloo_shimla_mirch": "aloo_group",
        "aloo_tikki": "aloo_group",
        "dum_aloo": "aloo_group",
        "butter_chicken": "chicken_group",
        "chicken_tikka": "chicken_group",
        "chicken_tikka_masala": "chicken_group",
        "chicken_razala": "chicken_group",
        "rasgulla": "sweet_white_group",
        "cham_cham": "sweet_white_group",
        "ras_malai": "sweet_white_group",
        "sandesh": "sweet_white_group",
        "chhena_kheeri": "sweet_white_group",
        "gulab_jamun": "sweet_brown_group",
        "jalebi": "sweet_brown_group",
        "imarti": "sweet_brown_group",
        "malapua": "sweet_brown_group",
        "double_ka_meetha": "sweet_brown_group",
        "adhirasam": "sweet_fried_group",
        "ariselu": "sweet_fried_group",
        "anarsa": "sweet_fried_group",
        "bandar_laddu": "sweet_fried_group",
        "unni_appam": "sweet_fried_group",
        "kajjikaya": "sweet_fried_group",
        "kakinada_khaja": "sweet_fried_group",
        "ghevar": "sweet_fried_group",
        "gavvalu": "sweet_fried_group",
        "roti": "bread_group",
        "chapati": "bread_group",
        "naan": "bread_group",
        "bhatura": "bread_group",
        "misi_roti": "bread_group",
        "makki_di_roti_sarson_da_saag": "bread_group",
        "dal_makhani": "dal_group",
        "dal_tadka": "dal_group",
        "chana_masala": "dal_group",
        "biryani": "rice_group",
        "daal_baati_churma": "rice_group",
    }
}

FOOD_DENSITY_DB = {
    "rice": 0.85,
    "dal": 1.05,
    "roti": 0.55,
    "paneer_curry": 1.10,
    "mixed_veg": 0.90,
    "aloo_gobi": 0.92,
    "default": 1.00
}
