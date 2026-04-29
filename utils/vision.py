import cv2
import numpy as np

def apply_nms(detections, iou_threshold):
    """
    Aplica Non-Maximum Suppression (NMS) para eliminar detecciones redundantes.

    Motivo: Los algoritmos de visión suelen encontrar múltiples cuadros superpuestos
    para un mismo objeto. NMS soluciona esto conservando solo el cuadro con mayor
    puntuación (score) y eliminando aquellos vecinos que tengan un alto solapamiento
    (Intersection over Union - IoU) con él.

    Referencia:
    - Explicación de NMS: https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    - Concepto de IoU: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    - OpenCV NMSBoxes: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee
    """
    if not detections:
        return []
    boxes  = [[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2,_ in detections]
    scores = [float(s) for *_,s in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    return [detections[i] for i in indices.flatten()]

# Suprime detecciones contenidas significativamente dentro de otras.
def suppress_contained(detections, containment_threshold, score_replacement_diff=0.12):
    """
    Elimina detecciones redundantes cuando una está contenida significativamente dentro de otra.
    
    Estrategia: Si un cuadro grande y uno pequeño se solapan casi totalmente, 
    preferimos el grande (el panel completo) a menos que el pequeño tenga una 
    nota mucho más alta (indicando que el grande es probablemente ruido).
    """
    if len(detections) <= 1:
        return detections

    # Ordenamos por score descendente (confianza)
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []

    for current in detections:
        x1, y1, x2, y2, score = current
        area = (x2 - x1) * (y2 - y1)
        if area <= 0: continue
        
        is_redundant = False
        to_replace = []

        for i, other in enumerate(kept):
            ox1, oy1, ox2, oy2, oscore = other
            oarea = (ox2 - ox1) * (oy2 - oy1)
            
            # Calcular intersección
            inter = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            if inter <= 0: continue

            # Ratios de contención
            contained_in_other = inter / area
            other_contained_in_this = inter / oarea

            # Caso A: La nueva detección está dentro de una que ya tenemos (y que tiene mejor nota)
            if contained_in_other > containment_threshold:
                is_redundant = True
                break
            
            # Caso B: La nueva detección "envuelve" a una que ya teníamos
            if other_contained_in_this > containment_threshold:
                # Si la diferencia de nota es pequeña, asumimos que el nuevo 
                # (grande) es el cartel completo y el viejo (pequeño) era un detalle/texto.
                if oscore - score < score_replacement_diff:
                    to_replace.append(i)
                else:
                    # Si el pequeño tenía mucha mejor nota, el grande es probablemente ruido.
                    is_redundant = True
                    break
        
        if not is_redundant:
            # Eliminar los trozos pequeños que han sido "absorbidos" por el nuevo cuadro grande
            for idx in sorted(to_replace, reverse=True):
                kept.pop(idx)
            kept.append(current)

    return kept

# Estima presencia de niebla basada en saturación y densidad de bordes.
def estimate_fog_conditions(hsv_image, gray_eq, fog_sat_thr, fog_edge_thr):
    """
    Estima presencia de niebla basada en saturación y densidad de bordes.
    
    Motivo: La niebla reduce drásticamente la saturación del color (los colores se ven grises/blancos)
    y difumina los contornos, reduciendo la densidad de bordes finos. Usamos Canny para medir 
    esta pérdida de altas frecuencias.
    
    Referencia:
    - Detección de bordes Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    - Efectos de la niebla en la imagen: https://en.wikipedia.org/wiki/Visibility
    """
    sat_mean = float(hsv_image[:, :, 1].mean())
    edges = cv2.Canny(gray_eq, 80, 160)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)
    is_fog = sat_mean < fog_sat_thr and edge_ratio < fog_edge_thr
    return is_fog

# Aplica CLAHE en espacio LAB para mejorar contraste local.
def enhance_image_clahe(image):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) en el canal de Luminancia (L) del espacio LAB.
    
    Motivo: A diferencia de la ecualización de histograma normal que actúa sobre toda la imagen, 
    CLAHE opera en pequeños bloques mejorando el contraste local sin amplificar en exceso 
    el ruido. Esto es vital para revelar paneles ocultos por mala iluminación.
    El espacio LAB se usa porque separa completamente la luminosidad (L) del color (A, B).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab_enh = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    enh_bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV)
    enh_gray_eq = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY)
    return enh_bgr, enh_hsv, enh_gray_eq


# Genera la máscara ideal para el score de correlación.
def build_ideal_mask(h, w, padding_ratio=0.08):
    """
    Genera la máscara ideal para el cálculo del score F1 de correlación.
    
    Motivo: Para verificar si un candidato es realmente un cartel rectangular, lo comparamos
    con una "plantilla ideal" (una matriz con 1s en el centro y 0s en un margen). Luego se calcula 
    el F1-Score (balance entre Precision y Recall) de la máscara detectada contra esta plantilla.
    
    Referencia:
    - F1-Score: https://en.wikipedia.org/wiki/F-score
    - Template Matching (concepto similar): https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    """
    mask = np.zeros((h, w), dtype=np.float32)
    bh, bw = int(h * padding_ratio), int(w * padding_ratio)
    mask[bh:h-bh, bw:w-bw] = 1.0
    return mask