LABEL = "multilabel"
SAMPLE_WEIGHTS = "sample_weights"
CRLF = "\n"

class_mapping_trustii = {
    0: "PNN",  # Polynucléaire Neutrophiles
    1: "LY",  # Lymphocytes
    2: "MO",  # Monocytes
    3: "EO",  # Polynucléaire Eosinophiles
    4: "BA",  # Polynucléaire Basophiles
    5: "Er",  # Erythroblaste
    6: "PM",  # Promyélocytes (ig)
    7: "Thromb",  # Plaquettes géantes
    8: "LAM3",  # Leucémie aigüe myéloïde 3
    9: "Lysee",  # Cellules lysées # Around 6xx*6xx
    10: "LyB",  # Lymphoblastes
    11: "LLC",  # LLC
    12: "MBL",  # Myéloblastes
    13: "LGL",  # Lymphocyte à grains
    14: "B",  # Blastes indifférenciés
    15: "M",  # Myélocytes (ig)
    16: "MM",  # Métamyélocytes (ig)
    17: "LF",  # Lymphome folliculaire
    18: "MoB",  # Monoblastes
    19: "LH_lyAct",  # Lymphocytes hyperbasophiles / activés
    20: "LM",  # Lymphome du manteau
    21: "LZMG",  # Lymphome de la zone marginale ganglionnaire
    22: "SS",  # Cellules de Sézary
}

class_mapping_trustii_ = {k: v + " (%d)" % k for k,v in class_mapping_trustii.items()}
class_labels_trustii = [str(k) for k, v in class_mapping_trustii.items()]
class_mapping_trustii_inv = {v: k for k, v in class_mapping_trustii.items()}
class_names_trustii = [str(v) for k, v in class_mapping_trustii.items()]

# Default mapping
class_mapping = class_mapping_trustii
class_mapping_ = class_mapping_trustii_
class_labels = class_labels_trustii
class_mapping_inv = class_mapping_trustii_inv
class_names = class_names_trustii
