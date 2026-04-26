# detector.py
# ============================================================================
# Módulo de detección de paneles informativos de autopista.
#
# Contiene la clase base abstracta Detector y la implementación DetectorBase
# que utiliza segmentación por color HSV + MSER + score por correlación + NMS.
#
# Asignatura: Visión Artificial – Universidad Rey Juan Carlos
#
# NOTA SOBRE CÓDIGO DE TERCEROS:
# Para la implementación de ciertas utilidades algorítmicas clásicas en visión
# por computador (como el cálculo IoU y NMS) se han adaptado ejemplos de
# tutoriales públicos (específicamente de PyImageSearch). La procedencia exacta
# se indica en los métodos correspondientes.
# ============================================================================

import cv2
import numpy as np


class Detector:
    """
    Clase base abstracta para detectores de paneles informativos.

    Define la interfaz común que deben implementar todos los detectores:
      - train():  carga/prepara datos de entrenamiento (opcional).
      - detect(): recibe una imagen BGR y devuelve lista de detecciones.
    """

    def __init__(self):
        pass

    def train(self, train_path: str):
        """
        Carga y prepara datos de entrenamiento si el detector los necesita.

        Args:
            train_path: ruta al directorio con imágenes y gt.txt de entrenamiento.
        """
        pass

    def detect(self, image: np.ndarray) -> list:
        """
        Detecta paneles informativos en una imagen.

        Args:
            image: imagen BGR (numpy array de shape HxWx3).

        Returns:
            Lista de tuplas (x1, y1, x2, y2, score) donde:
              - (x1, y1) es la esquina superior izquierda del bounding box.
              - (x2, y2) es la esquina inferior derecha.
              - score es un float entre 0 y 1 indicando confianza.
        """
        raise NotImplementedError("Las subclases deben implementar detect()")


class DetectorBase(Detector):
    """
    A) Segmentación por color HSV + cierre morfológico + contornos.
    B) MSER sobre imagen en escala de grises ecualizada.

    Cada candidato se evalúa con:
      1. Filtros de aspect ratio, tamaño y dimensión mínima.
      2. Proporción mínima de color azul saturado en espacio HSV.
      3. Score F1 entre la máscara azul del candidato y una máscara ideal.
      4. Verificación de borde claro/blanco alrededor de la región azul.
      5. NMS para eliminar detecciones redundantes.
    """

    def __init__(self):
        super().__init__()

        # --- Parámetros del detector MSER ---
        # 📎 Referencia parámetros MSER (indicada también en el enunciado):
        #    https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x
        # 📎 Ejemplo de uso oficial de MSER en OpenCV:
        #    https://github.com/opencv/opencv/blob/master/samples/python/mser.py
        self.mser = cv2.MSER_create()
        self.mser.setDelta(5)
        self.mser.setMinArea(600)
        self.mser.setMaxArea(100000)
        self.mser.setMaxVariation(0.25)
        self.mser.setMinDiversity(0.2)
        # Variante más sensible para escenas con niebla/bajo contraste
        self.mser_fog = cv2.MSER_create()
        self.mser_fog.setDelta(2)
        self.mser_fog.setMinArea(600)
        self.mser_fog.setMaxArea(100000)
        self.mser_fog.setMaxVariation(0.45)
        self.mser_fog.setMinDiversity(0.2)

        # --- Rangos HSV para azul saturado ---
        # 📎 Segmentación de color en espacio HSV con cv2.inRange:
        #    https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        # 📎 Cómo elegir los rangos HSV correctos para detección de color:
        #    https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-with-cv
        self.hsv_lower = np.array([100, 130, 80])
        self.hsv_upper = np.array([130, 255, 255])
        # Rango más permisivo para azul desaturado (niebla/humo)
        self.hsv_lower_fog = np.array([95, 45, 55])
        self.hsv_upper_fog = np.array([135, 255, 255])
        # Rango extra para niebla muy densa
        self.hsv_lower_dense_fog = np.array([95, 8, 80])
        self.hsv_upper_dense_fog = np.array([140, 255, 255])

        # --- Kernel para cierre morfológico ---
        # 📎 Operaciones morfológicas (erosión, dilatación, cierre) en OpenCV:
        #    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.morph_kernel_fog = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # --- Umbrales de filtrado ---
        self.min_blue_ratio = 0.35
        self.min_blue_ratio_fog = 0.22
        self.min_blue_ratio_dense_fog = 0.30
        self.min_blue_ratio_fog_fallback = 0.16
        self.min_aspect_ratio = 0.5
        self.max_aspect_ratio = 5.0
        self.min_bbox_area = 800
        self.max_bbox_area = 200000
        self.min_dimension = 30
        self.score_threshold = 0.55
        self.score_threshold_fog = 0.45
        self.score_threshold_dense_fog = 0.78
        self.score_threshold_fog_fallback = 0.48
        self.high_score_override = 0.92
        self.color_expand = 0.05
        self.mser_expand = 0.15

        # --- Verificación de borde claro ---
        self.border_margin = 6
        self.border_brightness_min = 90
        self.border_saturation_max = 140
        self.min_bright_borders = 1

        # --- NMS ---
        self.nms_iou_threshold = 0.45

        # --- Detección de niebla (heurística global) ---
        self.fog_sat_mean_max = 27.0
        self.dense_fog_sat_mean_max = 18.5
        self.fog_edge_ratio_max = 0.015

        # --- Filtros de posición y bordes ---
        self.bottom_roi_fraction = 0.86
        self.edge_score_min = 0.75
        self.containment_threshold = 0.65
        self.cluster_center_dist_factor = 0.28
        self.cluster_gap_factor = 0.40
        self.cluster_min_axis_overlap = 0.12

        # --- Máscara ideal para correlación ---
        # 📎 Concepto de template matching / correlación con máscara ideal:
        #    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
        self.mask_h = 40
        self.mask_w = 80
        self.ideal_mask = self._build_ideal_mask()

    # ======================================================================
    #  Métodos auxiliares de bajo nivel
    # ======================================================================

    def _build_ideal_mask(self) -> np.ndarray:
        """Máscara ideal del interior de un panel azul (40×80)."""
        mask = np.zeros((self.mask_h, self.mask_w), dtype=np.float32)
        bh = int(self.mask_h * 0.08)
        bw = int(self.mask_w * 0.08)
        mask[bh:self.mask_h - bh, bw:self.mask_w - bw] = 1.0
        return mask

    def _estimate_fog(self, hsv_image, gray_eq):
        """
        Estima si la imagen presenta niebla/baja visibilidad.
        Heurística propia basada en la media del canal S (saturación) en HSV
        y la densidad de bordes Canny: imágenes con niebla tienen baja saturación
        y pocos bordes definidos.
        """
        sat_mean = float(hsv_image[:, :, 1].mean())
        edges = cv2.Canny(gray_eq, 80, 160)
        edge_ratio = float(np.count_nonzero(edges) / edges.size)
        is_fog = sat_mean < self.fog_sat_mean_max and edge_ratio < self.fog_edge_ratio_max
        is_dense = sat_mean < self.dense_fog_sat_mean_max and edge_ratio < self.fog_edge_ratio_max
        return is_fog, is_dense

    def _get_blue_mask(self, roi_bgr, hsv_lower=None, hsv_upper=None):
        """
        Máscara binaria de píxeles azules, normalizada a tamaño fijo.
        📎 Conversión BGR→HSV y segmentación con inRange:
           https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        """
        hsv_lower = self.hsv_lower if hsv_lower is None else hsv_lower
        hsv_upper = self.hsv_upper if hsv_upper is None else hsv_upper
        roi = cv2.resize(roi_bgr, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return (cv2.inRange(hsv, hsv_lower, hsv_upper) / 255.0).astype(np.float32)

    def _compute_blue_ratio_hsv(self, roi_hsv, hsv_lower=None, hsv_upper=None):
        """Proporción de píxeles azules en una ROI HSV."""
        hsv_lower = self.hsv_lower if hsv_lower is None else hsv_lower
        hsv_upper = self.hsv_upper if hsv_upper is None else hsv_upper
        if roi_hsv.size == 0:
            return 0.0
        mask = cv2.inRange(roi_hsv, hsv_lower, hsv_upper)
        total = mask.shape[0] * mask.shape[1]
        return np.count_nonzero(mask) / total if total > 0 else 0.0

    def _compute_score(self, blue_mask):
        """
        Score F1 entre la máscara azul del candidato y la ideal.
        Combina precisión (píxeles azules bien colocados) y recall
        (cobertura del área esperada), siguiendo la métrica estándar F1.
        📎 Concepto de template matching con correlación elemento a elemento:
           https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
        """
        tp = np.sum(blue_mask * self.ideal_mask)
        recall = tp / (np.sum(self.ideal_mask) + 1e-6)
        precision = tp / (np.sum(blue_mask) + 1e-6)
        return float(2.0 * precision * recall / (precision + recall + 1e-6))

    def _check_bright_border(self, hsv_image, x1, y1, x2, y2):
        """Verifica si la región tiene bordes claros/blancos alrededor."""
        h_img, w_img = hsv_image.shape[:2]
        m = self.border_margin
        bright_count = 0
        total_checked = 0

        strips = [
            (y1 - m, y1, x1, x2, y1 - m >= 0),          # top
            (y2, y2 + m, x1, x2, y2 + m <= h_img),       # bottom
            (y1, y2, x1 - m, x1, x1 - m >= 0),           # left
            (y1, y2, x2, x2 + m, x2 + m <= w_img),       # right
        ]
        for r1, r2, c1, c2, valid in strips:
            if not valid:
                continue
            strip = hsv_image[r1:r2, c1:c2]
            if strip.size == 0:
                continue
            total_checked += 1
            if strip[:, :, 2].mean() > self.border_brightness_min and \
               strip[:, :, 1].mean() < self.border_saturation_max:
                bright_count += 1

        if total_checked <= 1:
            return True
        return bright_count >= self.min_bright_borders

    # ======================================================================
    #  NMS y filtros de post-procesamiento
    # ======================================================================

    def _compute_iou(self, boxA, boxB):
        """
        IoU entre dos bounding boxes.
        📎 Fuente/Inspiración: Algoritmo estándar de la literatura, adaptado
           del tutorial de PyImageSearch (Adrian Rosebrock).
           https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        """
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _nms(self, detections):
        """
        Non-Maximum Suppression: ordena por score y suprime IoU > umbral.
        📎 Fuente: Algoritmo clásico de NMS adaptado de tutoriales públicos
           (Malisiewicz et al. a través de PyImageSearch).
           https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
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
                if not suppressed[j] and self._compute_iou(det[:4], detections[j][:4]) > self.nms_iou_threshold:
                    suppressed[j] = True
        return kept

    def _suppress_contained(self, detections):
        """
        Suprime detecciones contenidas dentro de otra de mayor score.
        Solo si el contenedor tiene área < 8× la del contenido (evita
        suprimir paneles reales dentro de un box multi-panel).
        """
        if len(detections) <= 1:
            return detections
        detections = sorted(detections, key=lambda d: d[4], reverse=True)
        kept = []
        for det in detections:
            area_det = (det[2] - det[0]) * (det[3] - det[1])
            if area_det <= 0:
                continue
            is_contained = False
            for kd in kept:
                area_kd = (kd[2] - kd[0]) * (kd[3] - kd[1])
                if area_kd > 8.0 * area_det:
                    continue
                xA, yA = max(det[0], kd[0]), max(det[1], kd[1])
                xB, yB = min(det[2], kd[2]), min(det[3], kd[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                if inter / area_det > self.containment_threshold:
                    is_contained = True
                    break
            if not is_contained:
                kept.append(det)
        return kept

    def _axis_overlap_ratio(self, a1, a2, b1, b2):
        """Solape normalizado entre dos intervalos 1D."""
        overlap = max(0, min(a2, b2) - max(a1, b1))
        denom = max(1e-6, min(a2 - a1, b2 - b1))
        return overlap / denom

    def _should_cluster_pair(self, d1, d2):
        """
        Decide si dos detecciones cercanas deben agruparse en una envolvente.
        Criterio: centros próximos o gap reducido, con cierto alineamiento.
        """
        x1a, y1a, x2a, y2a = d1[:4]
        x1b, y1b, x2b, y2b = d2[:4]
        wa, ha = x2a - x1a, y2a - y1a
        wb, hb = x2b - x1b, y2b - y1b
        if wa <= 0 or ha <= 0 or wb <= 0 or hb <= 0:
            return False

        cxa, cya = (x1a + x2a) / 2.0, (y1a + y2a) / 2.0
        cxb, cyb = (x1b + x2b) / 2.0, (y1b + y2b) / 2.0
        dx = abs(cxa - cxb)
        dy = abs(cya - cyb)
        ref_w = max(wa, wb)
        ref_h = max(ha, hb)
        x_overlap = self._axis_overlap_ratio(x1a, x2a, x1b, x2b)
        y_overlap = self._axis_overlap_ratio(y1a, y2a, y1b, y2b)

        close_centers = (
            dx <= self.cluster_center_dist_factor * ref_w and
            dy <= self.cluster_center_dist_factor * ref_h
        )

        # Gap entre cajas en ejes horizontal/vertical.
        h_gap = max(0, max(x1a, x1b) - min(x2a, x2b))
        v_gap = max(0, max(y1a, y1b) - min(y2a, y2b))
        close_gap = (
            (h_gap <= self.cluster_gap_factor * ref_w and
             y_overlap >= self.cluster_min_axis_overlap) or
            (v_gap <= self.cluster_gap_factor * ref_h and
             x_overlap >= self.cluster_min_axis_overlap)
        )
        return close_centers or close_gap

    def _cluster_by_proximity(self, detections):
        """
        Agrupa detecciones próximas y las fusiona en un rectángulo envolvente.
        """
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
                    if visited[v]:
                        continue
                    if self._should_cluster_pair(detections[u], detections[v]):
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

    def _filter_by_position(self, detections, h_img, w_img):
        """Filtra detecciones en posiciones inverosímiles (asfalto, esquinas)."""
        margin = 5
        filtered = []
        for det in detections:
            x1, y1, x2, y2, score = det
            # Centro en zona inferior (carretera/capó)
            if (y1 + y2) / 2 > h_img * self.bottom_roi_fraction:
                continue
            # Falsos positivos típicos: franjas bajas azules (capó/carrocería).
            h_box = y2 - y1
            if y1 > h_img * 0.50 and h_box < h_img * 0.08 and score < 0.82:
                continue
            # Toca ≥2 bordes de imagen (esquinas/bandas)
            touches = sum([x1 <= margin, x2 >= w_img - margin,
                           y1 <= margin, y2 >= h_img - margin])
            if touches >= 2:
                continue
            filtered.append(det)
        return filtered

    def _filter_edge_low_score(self, detections, h_img, w_img):
        """Filtra detecciones en borde de imagen con score bajo."""
        margin = 5
        return [d for d in detections
                if not ((d[0] <= margin or d[2] >= w_img - margin or
                         d[1] <= margin or d[3] >= h_img - margin)
                        and d[4] < self.edge_score_min)]

    def _postprocess(self, detections, h_img, w_img):
        """Aplica todos los filtros de post-procesamiento."""
        detections = self._cluster_by_proximity(detections)
        detections = self._nms(detections)
        detections = self._suppress_contained(detections)
        detections = self._filter_by_position(detections, h_img, w_img)
        detections = self._filter_edge_low_score(detections, h_img, w_img)
        return detections

    # ======================================================================
    #  Generación y evaluación de candidatos
    # ======================================================================

    def _get_color_candidates(self, hsv_image, hsv_lower, hsv_upper,
                              use_small_kernel=False):
        """
        Candidatos por segmentación de color HSV + cierre + contornos.
        📎 Operación de cierre morfológico para unir regiones fragmentadas:
           https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        """
        mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
        kernel = self.morph_kernel_fog if use_small_kernel else self.morph_kernel
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours
                if cv2.boundingRect(c)[2] > 0 and cv2.boundingRect(c)[3] > 0]

    def _get_mser_candidates(self, gray_eq, use_fog_mser=False):
        """
        Candidatos por MSER sobre imagen ecualizada.
        📎 Parámetros MSER en OpenCV:
           https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x
        📎 Ejemplo oficial de uso:
           https://github.com/opencv/opencv/blob/master/samples/python/mser.py
        """
        engine = self.mser_fog if use_fog_mser else self.mser
        regions, _ = engine.detectRegions(gray_eq)
        return [cv2.boundingRect(r) for r in regions
                if cv2.boundingRect(r)[2] > 0 and cv2.boundingRect(r)[3] > 0]

    def _evaluate_candidate(self, image, hsv_image,
                            x, y, w, h, expand,
                            hsv_lower, hsv_upper,
                            min_blue_ratio, score_threshold=None,
                            require_border=True):
        """
        Evalúa un candidato: filtros geométricos + color + score + borde.
        Returns (x1,y1,x2,y2,score) o None.
        """
        h_img, w_img = image.shape[:2]
        aspect = w / h
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return None
        area = w * h
        if area < self.min_bbox_area or area > self.max_bbox_area:
            return None
        if w < self.min_dimension or h < self.min_dimension:
            return None

        # Expandir bbox
        pad_x, pad_y = int(w * expand), int(h * expand)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Filtro de proporción azul
        blue_ratio = self._compute_blue_ratio_hsv(
            hsv_image[y1:y2, x1:x2], hsv_lower, hsv_upper)
        if blue_ratio < min_blue_ratio:
            return None

        # Score F1
        score = self._compute_score(self._get_blue_mask(roi, hsv_lower, hsv_upper))
        thr = self.score_threshold if score_threshold is None else score_threshold
        if score < thr:
            return None

        # Borde claro
        if require_border:
            # Comprobar en bbox original y expandido
            ox1, oy1 = max(0, x), max(0, y)
            ox2, oy2 = min(w_img, x + w), min(h_img, y + h)
            border_ok = (self._check_bright_border(hsv_image, ox1, oy1, ox2, oy2) or
                         self._check_bright_border(hsv_image, x1, y1, x2, y2))
            if not border_ok and score < self.high_score_override:
                return None

        return (x1, y1, x2, y2, score)

    def _run_pass(self, image, hsv_image, gray_eq, pass_configs):
        """
        Ejecuta un pase de detección con las configuraciones dadas.

        Args:
            pass_configs: lista de dicts con claves:
                color_hsv_lo, color_hsv_hi, small_kernel,
                use_fog_mser, expand_color, expand_mser,
                hsv_lower, hsv_upper, min_blue_ratio,
                score_threshold, require_border
        Returns:
            Lista de candidatos (x1,y1,x2,y2,score).
        """
        candidates = []
        seen = set()
        for cfg in pass_configs:
            # Candidatos por color
            color_rects = self._get_color_candidates(
                hsv_image, cfg["color_hsv_lo"], cfg["color_hsv_hi"],
                use_small_kernel=cfg.get("small_kernel", False))
            # Candidatos por MSER
            mser_rects = self._get_mser_candidates(
                gray_eq, use_fog_mser=cfg.get("use_fog_mser", False))

            for rect_list, expand in [(color_rects, cfg.get("expand_color", self.color_expand)),
                                      (mser_rects, cfg.get("expand_mser", self.mser_expand))]:
                for rect in rect_list:
                    if rect in seen:
                        continue
                    seen.add(rect)
                    result = self._evaluate_candidate(
                        image, hsv_image, *rect, expand=expand,
                        hsv_lower=cfg["hsv_lower"], hsv_upper=cfg["hsv_upper"],
                        min_blue_ratio=cfg["min_blue_ratio"],
                        score_threshold=cfg.get("score_threshold"),
                        require_border=cfg.get("require_border", True))
                    if result is not None:
                        candidates.append(result)
        return candidates

    def _enhance_for_dense_fog(self, image):
        """
        Mejora contraste local con CLAHE en LAB para recuperar
        paneles azules en niebla muy densa.
        📎 Fuente: Técnica estándar recomendada en la documentación oficial de OpenCV
           (equalización de histograma y CLAHE):
           https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        📎 Aumento de contraste en imágenes con OpenCV en Python:
           https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        Returns (enhanced_bgr, enhanced_hsv, enhanced_gray_eq).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab_enh = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
        enh_bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
        enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV)
        enh_gray_eq = cv2.equalizeHist(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY))
        return enh_bgr, enh_hsv, enh_gray_eq

    # ======================================================================
    #  Detección principal
    # ======================================================================

    def detect(self, image: np.ndarray) -> list:
        """
        Detecta paneles informativos azules en una imagen.

        Pipeline:
          1. Pase normal (HSV estricto).
          2. Pase niebla (HSV relajado, solo si hay niebla).
          3. NMS.
          4. Pase niebla densa + CLAHE (solo si vacío y niebla densa).
          5. Pase fallback sin borde (solo si vacío y niebla).
          6. Post-procesamiento (contención + posición + bordes).
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_eq = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        is_fog, is_dense_fog = self._estimate_fog(hsv_image, gray_eq)
        h_img, w_img = image.shape[:2]

        # Configuraciones de pases
        cfg_normal = {
            "color_hsv_lo": self.hsv_lower, "color_hsv_hi": self.hsv_upper,
            "hsv_lower": self.hsv_lower, "hsv_upper": self.hsv_upper,
            "min_blue_ratio": self.min_blue_ratio,
            "score_threshold": self.score_threshold,
        }
        cfg_fog = {
            "color_hsv_lo": self.hsv_lower_fog, "color_hsv_hi": self.hsv_upper_fog,
            "hsv_lower": self.hsv_lower_fog, "hsv_upper": self.hsv_upper_fog,
            "min_blue_ratio": self.min_blue_ratio_fog,
            "score_threshold": self.score_threshold_fog,
            "small_kernel": True, "use_fog_mser": True,
        }

        # Pase 0: normal + (opcionalmente) niebla
        passes = [cfg_normal]
        if is_fog:
            passes.append(cfg_fog)
        candidates = self._run_pass(image, hsv_image, gray_eq, passes)
        final = self._nms(candidates)

        # Pase 2: niebla densa con CLAHE (solo si vacío)
        if is_dense_fog and not final:
            enh_bgr, enh_hsv, enh_gray_eq = self._enhance_for_dense_fog(image)
            cfg_dense = {
                "color_hsv_lo": self.hsv_lower_dense_fog,
                "color_hsv_hi": self.hsv_upper_dense_fog,
                "hsv_lower": self.hsv_lower_dense_fog,
                "hsv_upper": self.hsv_upper_dense_fog,
                "min_blue_ratio": self.min_blue_ratio_dense_fog,
                "score_threshold": self.score_threshold_dense_fog,
                "small_kernel": True, "use_fog_mser": True,
            }
            cfg_fog_enh = {**cfg_fog}  # También fog range sobre CLAHE
            dense_cands = self._run_pass(
                enh_bgr, enh_hsv, enh_gray_eq, [cfg_dense, cfg_fog_enh])
            dense_nms = self._nms(dense_cands)
            if len(dense_nms) <= 8:  # Sanity: demasiadas = ruido
                final = dense_nms

        # Pase fallback: niebla sin detecciones, relajar umbrales
        if is_fog and not final:
            cfg_fallback = {
                "color_hsv_lo": self.hsv_lower_fog,
                "color_hsv_hi": self.hsv_upper_fog,
                "hsv_lower": self.hsv_lower_fog,
                "hsv_upper": self.hsv_upper_fog,
                "min_blue_ratio": self.min_blue_ratio_fog_fallback,
                "score_threshold": self.score_threshold_fog_fallback,
                "small_kernel": True, "use_fog_mser": True,
                "require_border": False,
            }
            fallback = self._nms(
                self._run_pass(image, hsv_image, gray_eq, [cfg_fallback]))
            if len(fallback) <= 8:
                final = fallback

        return self._postprocess(final, h_img, w_img)

    # ======================================================================
    #  Debug (para debug_pipeline.py)
    # ======================================================================

    def debug_detect(self, image: np.ndarray) -> tuple:
        """
        Ejecuta detect() y devuelve también las detecciones.
        Versión simplificada que reutiliza detect().
        Returns (detections, []).
        """
        return self.detect(image), []


class DetectorHough(DetectorBase):
    """
    Detector Alternativo para el Punto 4 (Transformada de Hough).

    Utiliza cv2.HoughLinesP para extraer bordes rectos (segmentos de línea).
    A partir de estos segmentos, reconstruye un esqueleto de líneas que se intersectan
    (posibles marcos rectangulares de paneles) de manera muy eficiente (O(N)), y
    busca contornos sobre dicha estructura para determinar los candidatos a panel.

    El score se calcula mediante F1 entre la máscara azul del candidato y la máscara
    ideal, igual que en DetectorBase (coherencia entre detectores).

    📎 Transformada de Hough Probabilística en OpenCV:
       https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
    📎 Uso de HoughLinesP para detectar segmentos rectos:
       https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    """

    def __init__(self):
        super().__init__()

    def detect(self, image: np.ndarray) -> list:
        h_img, w_img = image.shape[:2]

        # 1. Convertir a escala de grises y suavizar para reducir ruido
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Detector de bordes Canny
        edges = cv2.Canny(blur, 50, 150)

        # 3. Transformada de Hough Probabilística para extraer segmentos rectos
        # 📎 https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40,
                                minLineLength=25, maxLineGap=15)

        candidates = []

        if lines is not None:
            # 4. Dibujar los segmentos de Hough en una máscara
            hough_mask = np.zeros_like(edges)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 3)

            # 5. Extraer contornos de las intersecciones de Hough
            contours, _ = cv2.findContours(hough_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h + 1e-5)
                area = w * h

                # 6. Filtrado geométrico básico
                if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio and
                        self.min_bbox_area <= area <= self.max_bbox_area and
                        w >= self.min_dimension and h >= self.min_dimension):
                    continue

                # 7. Filtro de proporción azul (permisivo, rango fog)
                # 📎 Segmentación HSV con inRange:
                #    https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
                roi_hsv = hsv_image[y:y+h, x:x+w]
                blue_ratio = self._compute_blue_ratio_hsv(
                    roi_hsv, self.hsv_lower_fog, self.hsv_upper_fog)
                if blue_ratio < 0.15:
                    continue

                # 8. Expandir bbox
                pad_x, pad_y = int(w * 0.05), int(h * 0.05)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + w + pad_x)
                y2 = min(h_img, y + h + pad_y)

                roi_bgr = image[y1:y2, x1:x2]
                if roi_bgr.size == 0:
                    continue

                # 9. Score F1 igual que DetectorBase (consistencia entre detectores)
                # 📎 https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
                blue_mask = self._get_blue_mask(roi_bgr, self.hsv_lower_fog,
                                                self.hsv_upper_fog)
                score = self._compute_score(blue_mask)

                if score < self.score_threshold_fog:
                    continue

                candidates.append((x1, y1, x2, y2, float(score)))

        # 10. Post-procesamiento estándar
        # 📎 NMS adaptado de PyImageSearch:
        #    https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        final = self._nms(candidates)
        return self._postprocess(final, h_img, w_img)