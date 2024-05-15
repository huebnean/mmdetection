# Bibliothek enthält Funktionen zum berechnen des Intersection over Unions Wertes sowie Funktionen, die diese Werte zur Zuordnung von Labels nutzen

def intersection_over_union(bb_a, bb_b):
    '''
    Funktion berechnet den IoU Wert
    
    INPUT:
    bb_a: Bounding-Box Bild A -> Liste [x_1, y_1, x_2, y_2]
    bb_b: Bounding-Box Bild B -> Liste [x_1, y_1, x_2, y_2]

    OUTPUT:
    IoU Wert
    '''
    # berechne die Koordinaten der Intersection
    bb_intersection = get_intersection_coordinates(bb_a, bb_b)

    # Wenn es keine Intersection gibt
    if(not bb_intersection):
        # Gebe 0 zurück
        return 0

    # Fläche der Intersection Area berechnen
    intersection_area = calc_bb_area(bb_intersection)
    
    # Fläche der Union Area berechnen
    union_area = calc_bb_area(bb_a) + calc_bb_area(bb_b) - intersection_area
    if union_area != 0:
        return intersection_area/union_area
    else:
        return 0

def get_specific_patch_intersection_over_union(bbox_a, bbox_b, bbox_patch):
    '''
    This function calculates the iou based on the adjusted size of the bboxes by a patch

    INPUT:
    bbox_a: coordinates of the bbox a (x0,y0,x1,y1)
    bbox_b: coordinates of the bbox b (x0,y0,x1,y1)
    bbox_patch: coordinates of the patch (x0,y0,x1,y1)

    OUTPUT:
    value of the specific "intersection over union" (adjusted by the patch)
    '''

    # calculate intersection of all bboxes with each other
    bbox_intersection_a_b = get_intersection_coordinates(bbox_a, bbox_b)
    bbox_intersection_patch_a = get_intersection_coordinates(bbox_patch, bbox_a)
    bbox_intersection_patch_b = get_intersection_coordinates(bbox_patch, bbox_b)

    # calculate coordinates of the intersection bbox between patch and the intersection of a and b
    bbox_intersection_patch_inter_a_b = get_intersection_coordinates(bbox_patch, bbox_intersection_a_b)

    # size of the intersection area (patch is substracted of original intersection of a and b)
    area_adjusted_intersection = calc_bb_area(bbox_intersection_a_b) - calc_bb_area(bbox_intersection_patch_inter_a_b)

    # size of the union area (patch is substracted of original union of a and b)
    area_adjusted_union = calc_bb_area(bbox_a) - calc_bb_area(bbox_intersection_patch_a) + calc_bb_area(bbox_b) - calc_bb_area(bbox_intersection_patch_b) - area_adjusted_intersection

    if area_adjusted_union != 0:
        return area_adjusted_intersection / area_adjusted_union 
    else:
        return 0

def calc_bb_area(bb):
    '''
    Funktion berechnet aus Start und Endwert die Fläche einer Bounding Box
    '''

    return (bb[2] - bb[0]) * (bb[3] - bb[1])

def get_intersection_coordinates(bb_a, bb_b):
    '''
    Funktion berechnet die Koordinaten der Intersection 
    
    INPUT:
    bb_a: Bounding-Box Bild A -> Liste [x_1, y_1, x_2, y_2]
    bb_b: Bounding-Box Bild B -> Liste [x_1, y_1, x_2, y_2]

    OUTPUT:
    Intersection Koordinaten
    '''

    # Falls keine Überschneidung existiert -> False zurückgeben
    if(not have_intersection(bb_a, bb_b)):
        return (0, 0, 0, 0)

    # obere linke Ecke der intersection Area finden
    bb_intersec_ul_x = max(bb_a[0], bb_b[0])
    bb_intersec_ul_y = max(bb_a[1], bb_b[1])
    # untere rechte Ecke der intersection Area finden
    bb_intersec_lr_x = min(bb_a[2], bb_b[2])
    bb_intersec_lr_y = min(bb_a[3], bb_b[3])

    return (bb_intersec_ul_x, bb_intersec_ul_y ,bb_intersec_lr_x, bb_intersec_lr_y)

def have_intersection(bb_a, bb_b):
    '''
    Funktion überprüft ob zwei Bounding boxen eine Schnittmenge beszitzen
    '''

    # Überprüfe ob eines der Rechtecke eine Linie ist
    if(bb_a[0] == bb_a[2] or bb_a[1] == bb_a[3] or bb_b[0] == bb_b[2] or bb_b[1] == bb_b[3]):
        return False
    
    # überprüfe, ob eines der Rechtecke links vom Anderen ist
    if(bb_a[2] <= bb_b[0] or bb_b[2] <= bb_a[0]):
        return False

    # überprüfe, ob eines der Rechtecke über dem Anderen ist
    if(bb_a[3] <= bb_b[1] or bb_b[3] <= bb_a[1]):
        return False

    return True