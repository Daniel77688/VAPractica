import cv2
import numpy as np

# Calcula el Intersection over Union (IoU) entre dos bounding boxes.
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (area_a + area_b - inter + 1e-6)

# Aplica Non-Maximum Suppression a una lista de detecciones.
# detections: lista de (x1, y1, x2, y2, score)
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
    """
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    suppressed = [False] * len(detections)
    for i, det in enumerate(detections):
        if suppressed[i]:
            continue
        kept.append(det)
        for j in range(i + 1, len(detections)):
            if not suppressed[j] and compute_iou(det[:4], detections[j][:4]) > iou_threshold:
                suppressed[j] = True
    return kept

# Determina si dos detecciones se tocan o solapan.
def should_cluster_pair(d1, d2):
    x1a, y1a, x2a, y2a = d1[:4]
    x1b, y1b, x2b, y2b = d2[:4]
    xA, yA = max(x1a, x1b), max(y1a, y1b)
    xB, yB = min(x2a, x2b), min(y2a, y2b)
    return (xB >= xA) and (yB >= yA)

# Agrupa detecciones próximas y las fusiona en su rectángulo envolvente.
def cluster_by_proximity(detections):
    n = len(detections)
    if n <= 1:
        return detections
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in range(n):
                if not visited[v] and should_cluster_pair(detections[u], detections[v]):
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        clusters.append(comp)

    merged = []
    for comp in clusters:
        if len(comp) == 1:
            merged.append(detections[comp[0]])
            continue
        boxes = [detections[idx] for idx in comp]
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        score = max(b[4] for b in boxes)
        merged.append((x1, y1, x2, y2, score))
    return merged

# Suprime detecciones contenidas significativamente dentro de otras.
def suppress_contained(detections, containment_threshold):
    if len(detections) <= 1:
        return detections
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    for det in detections:
        area_det = (det[2] - det[0]) * (det[3] - det[1])
        if area_det <= 0: continue
        is_contained = False
        for kd in kept:
            area_kd = (kd[2] - kd[0]) * (kd[3] - kd[1])
            xA, yA = max(det[0], kd[0]), max(det[1], kd[1])
            xB, yB = min(det[2], kd[2]), min(det[3], kd[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if (inter / area_det > containment_threshold) or (inter / area_kd > containment_threshold):
                is_contained = True
                break
        if not is_contained:
            kept.append(det)
    return kept

# Estima presencia de niebla y niebla densa basada en saturación y densidad de bordes.
def estimate_fog_conditions(hsv_image, gray_eq, fog_sat_thr, fog_edge_thr, dense_sat_thr):
    """
    Estima presencia de niebla y niebla densa basada en saturación y densidad de bordes.
    
    Motivo: La niebla reduce drásticamente la saturación del color (los colores se ven grises/blancos)
    y difumina los contornos, reduciendo la densidad de bordes finos. Usamos Canny para medir 
    esta pérdida de altas frecuencias.
    
    Referencia:
    - Detección de bordes Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    - Efectos de la niebla en la imagen: https://en.wikipedia.org/wiki/Visibility_(geometry)
    """
    sat_mean = float(hsv_image[:, :, 1].mean())
    edges = cv2.Canny(gray_eq, 80, 160)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)
    is_fog = sat_mean < fog_sat_thr and edge_ratio < fog_edge_thr
    is_dense = sat_mean < dense_sat_thr and edge_ratio < fog_edge_thr
    return is_fog, is_dense

# Aplica CLAHE en espacio LAB para mejorar contraste local.
def enhance_image_clahe(image):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) en el canal de Luminancia (L) del espacio LAB.
    
    Motivo: A diferencia de la ecualización de histograma normal que actúa sobre toda la imagen, 
    CLAHE opera en pequeños bloques mejorando el contraste local sin amplificar en exceso 
    el ruido. Esto es vital para revelar paneles ocultos por niebla o mala iluminación.
    El espacio LAB se usa porque separa completamente la luminosidad (L) del color (A, B).
    
    Referencia: 
    - OpenCV CLAHE: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    - Espacio de color LAB: https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab_enh = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    enh_bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV)
    enh_gray_eq = cv2.equalizeHist(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY))
    return enh_bgr, enh_hsv, enh_gray_eq

# Genera la máscara ideal para el score de correlación.
def build_ideal_mask(h, w, padding_ratio=0.08):
    """
    Genera la máscara ideal para el cálculo del score F1 de correlación.
    
    Motivo: Para verificar si un candidato es realmente un cartel rectangular, lo comparamos
    con una "plantilla ideal" (una matriz con 1s en el centro y 0s en un pequeño margen).
    Luego se calcula el F1-Score (balance entre Precision y Recall) de la máscara detectada 
    contra esta plantilla.
    
    Referencia:
    - F1-Score: https://en.wikipedia.org/wiki/F-score
    - Template Matching (concepto similar): https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    """
    mask = np.zeros((h, w), dtype=np.float32)
    bh, bw = int(h * padding_ratio), int(w * padding_ratio)
    mask[bh:h-bh, bw:w-bw] = 1.0
    return mask