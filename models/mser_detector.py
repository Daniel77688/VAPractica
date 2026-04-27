import cv2
import numpy as np
from .interface import Detector
from utils.vision import (
    apply_nms, cluster_by_proximity, suppress_contained, 
    estimate_fog_conditions, enhance_image_clahe, build_ideal_mask
)

class DetectorBase(Detector):
    def __init__(self):
        super().__init__()
        # Configuración MSER
        self.mser = cv2.MSER_create(delta=3, min_area=600, max_area=100000, max_variation=0.25, min_diversity=0.2)
        self.mser_fog = cv2.MSER_create(delta=2, min_area=600, max_area=100000, max_variation=0.45, min_diversity=0.2)

        # Rangos HSV
        self.hsv_lower = np.array([100, 130, 80])
        self.hsv_upper = np.array([130, 255, 255])
        self.hsv_lower_fog = np.array([95, 45, 55])
        self.hsv_upper_fog = np.array([135, 255, 255])
        self.hsv_lower_dense_fog = np.array([95, 8, 80])
        self.hsv_upper_dense_fog = np.array([140, 255, 255])

        # Parámetros y Umbrales
        self.min_blue_ratio = 0.50
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
        
        self.color_expand = 0.02
        self.mser_expand = 0.02
        
        # Post-procesamiento
        self.nms_iou_threshold = 0.45
        self.containment_threshold = 0.65
        self.bottom_roi_fraction = 0.98
        self.edge_score_min = 0.75
        
        # Borde claro
        self.border_margin = 6
        self.border_brightness_min = 90
        self.border_saturation_max = 140
        self.min_bright_borders = 1
        
        # Heurística niebla
        self.fog_sat_mean_max = 27.0
        self.dense_fog_sat_mean_max = 18.5
        self.fog_edge_ratio_max = 0.015

        # Máscara ideal
        self.mask_h, self.mask_w = 40, 80
        self.ideal_mask = build_ideal_mask(self.mask_h, self.mask_w)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.morph_kernel_fog = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Verifica que la geometría de una bounding box sea válida
    @staticmethod
    def _is_valid_geometry(w, h, min_aspect_ratio, max_aspect_ratio, min_bbox_area, max_bbox_area, min_dimension):
        if h <= 0:
            return False
        aspect = w / h
        area = w * h
        return (
            min_aspect_ratio <= aspect <= max_aspect_ratio
            and min_bbox_area <= area <= max_bbox_area
            and w >= min_dimension
            and h >= min_dimension
        )

    # Método principal de detección
    def detect(self, image: np.ndarray) -> list:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_eq = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        is_fog, is_dense_fog = estimate_fog_conditions(
            hsv_image, gray_eq, self.fog_sat_mean_max, self.fog_edge_ratio_max, self.dense_fog_sat_mean_max
        )
        h_img, w_img = image.shape[:2]

        # Pases de detección
        cfg_normal = self._create_cfg(self.hsv_lower, self.hsv_upper, self.min_blue_ratio, self.score_threshold)
        cfg_fog = self._create_cfg(self.hsv_lower_fog, self.hsv_upper_fog, self.min_blue_ratio_fog, self.score_threshold_fog, small=True, fog_mser=True)

        passes = [cfg_normal]
        if is_fog: passes.append(cfg_fog)
        
        candidates = self._run_pass(image, hsv_image, gray_eq, passes)
        final = apply_nms(candidates, self.nms_iou_threshold)

        # Pases extra si vacío
        if is_dense_fog and not final:
            enh_bgr, enh_hsv, enh_gray_eq = enhance_image_clahe(image)
            cfg_dense = self._create_cfg(self.hsv_lower_dense_fog, self.hsv_upper_dense_fog, self.min_blue_ratio_dense_fog, self.score_threshold_dense_fog, small=True, fog_mser=True)
            dense_cands = self._run_pass(enh_bgr, enh_hsv, enh_gray_eq, [cfg_dense, cfg_fog])
            final = apply_nms(dense_cands, self.nms_iou_threshold)

        if is_fog and not final:
            cfg_fallback = self._create_cfg(self.hsv_lower_fog, self.hsv_upper_fog, self.min_blue_ratio_fog_fallback, self.score_threshold_fog_fallback, small=True, fog_mser=True, border=False)
            final = apply_nms(self._run_pass(image, hsv_image, gray_eq, [cfg_fallback]), self.nms_iou_threshold)

        # En escenas muy ruidosas (p.ej. vegetación/patrones), forzamos una poda extra.
        if len(final) >= 20:
            final = [d for d in final if d[4] >= 0.90]
        return self._postprocess(final, h_img, w_img)

    # Crea una configuración de detección
    def _create_cfg(self, lo, hi, ratio, thr, small=False, fog_mser=False, border=True):
        return {
            "hsv_lower": lo, "hsv_upper": hi,
            "min_blue_ratio": ratio, "score_threshold": thr, "small_kernel": small, 
            "use_fog_mser": fog_mser, "require_border": border
        }

    # Ejecuta un pase de detección
    def _run_pass(self, image, hsv_image, gray_eq, pass_configs):
        candidates = []
        seen = set()
        color_cache = {}
        mser_cache = {}
        for cfg in pass_configs:
            color_key = (
                tuple(cfg["hsv_lower"].tolist()),
                tuple(cfg["hsv_upper"].tolist()),
                cfg.get("small_kernel", False),
            )
            if color_key not in color_cache:
                color_cache[color_key] = self._get_color_candidates(
                    hsv_image, cfg["hsv_lower"], cfg["hsv_upper"], cfg.get("small_kernel", False)
                )
            color_rects = color_cache[color_key]

            mser_key = cfg.get("use_fog_mser", False)
            if mser_key not in mser_cache:
                mser_cache[mser_key] = self._get_mser_candidates(gray_eq, mser_key)
            mser_rects = mser_cache[mser_key]

            for rect_list, expand in [(color_rects, self.color_expand), (mser_rects, self.mser_expand)]:
                for rect in rect_list:
                    if rect in seen: continue
                    seen.add(rect)
                    res = self._evaluate_candidate(image, hsv_image, *rect, expand, cfg)
                    if res: candidates.append(res)
        return candidates


    # Evalúa un candidato
    def _evaluate_candidate(self, image, hsv_image, x, y, w, h, expand, cfg):
        h_img, w_img = image.shape[:2]
        if not self._is_valid_geometry(
            w,
            h,
            self.min_aspect_ratio,
            self.max_aspect_ratio,
            self.min_bbox_area,
            self.max_bbox_area,
            self.min_dimension,
        ):
            return None

        pad_x, pad_y = int(w * expand), int(h * expand)
        x1, y1, x2, y2 = max(0, x-pad_x), max(0, y-pad_y), min(w_img, x+w+pad_x), min(h_img, y+h+pad_y)
        
        roi_hsv = hsv_image[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            return None
        
        if self._compute_blue_ratio(roi_hsv, cfg["hsv_lower"], cfg["hsv_upper"]) < cfg["min_blue_ratio"]:
            return None

        score = self._compute_f1_score(image[y1:y2, x1:x2], cfg["hsv_lower"], cfg["hsv_upper"])
        if score < cfg["score_threshold"]:
            return None

        if cfg.get("require_border", True):
            if not (self._check_border(hsv_image, x, y, x+w, y+h) or self._check_border(hsv_image, x1, y1, x2, y2)):
                if score < self.high_score_override:
                    return None

        return (x1, y1, x2, y2, score)

    # Calcula la proporción de azul en una región de interés (ROI)
    def _compute_blue_ratio(self, roi_hsv, lo, hi):
        mask = cv2.inRange(roi_hsv, lo, hi)
        return np.count_nonzero(mask) / mask.size if mask.size > 0 else 0


    # Calcula el score F1 basado en la máscara ideal.
    def _compute_f1_score(self, roi_bgr, lo, hi):
        """
        Calcula el F1-Score basado en la correlación geométrica con la máscara ideal.
        
        Motivo: El F1-Score es la media armónica entre Precision (cuántos píxeles detectados son
        realmente parte del cartel) y Recall (cuántos píxeles del cartel logramos detectar).
        Esto penaliza candidatos que tienen la forma incorrecta (ej. líneas delgadas o círculos)
        incluso si son del color correcto.
        
        Referencia:
        - F1-Score: https://en.wikipedia.org/wiki/F-score
        """
        roi = cv2.resize(roi_bgr, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)
        mask = (cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), lo, hi) / 255.0).astype(np.float32)
        tp = np.sum(mask * self.ideal_mask)
        recall = tp / (np.sum(self.ideal_mask) + 1e-6)
        prec = tp / (np.sum(mask) + 1e-6)
        return float(2.0 * prec * recall / (prec + recall + 1e-6))


    # Verifica la presencia de un borde claro alrededor de una región
    def _check_border(self, hsv, x1, y1, x2, y2):
        h_img, w_img = hsv.shape[:2]
        m = self.border_margin
        strips = [(y1-m, y1, x1, x2, y1-m>=0), (y2, y2+m, x1, x2, y2+m<=h_img), (y1, y2, x1-m, x1, x1-m>=0), (y1, y2, x2, x2+m, x2+m<=w_img)]
        bright = 0
        total = 0
        for r1, r2, c1, c2, valid in strips:
            if not valid: continue
            s = hsv[r1:r2, c1:c2]
            if s.size == 0: continue
            total += 1
            if s[:,:,2].mean() > self.border_brightness_min and s[:,:,1].mean() < self.border_saturation_max:
                bright += 1
        return True if total <= 1 else bright >= self.min_bright_borders

    # Obtiene candidatos de color
    def _get_color_candidates(self, hsv, lo, hi, small):
        """
        Obtiene regiones candidatas filtrando por color y aplicando operaciones morfológicas.
        
        Motivo: Al aislar el rango de color (ej. azul), nos quedamos solo con los píxeles de interés.
        La operación de cierre morfológico (dilatación seguida de erosión) sirve para fusionar
        píxeles cercanos y rellenar pequeños agujeros, creando bloques sólidos que luego se extraen
        con findContours.
        
        Referencia:
        - Espacio HSV y filtrado inRange: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        - Operaciones Morfológicas: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        """
        mask = cv2.inRange(hsv, lo, hi)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel_fog if small else self.morph_kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours]

    # Obtiene candidatos de MSER
    def _get_mser_candidates(self, gray, fog):
        """
        Obtiene regiones candidatas usando el algoritmo MSER.
        
        Motivo: Maximally Stable Extremal Regions (MSER) es excelente para encontrar regiones
        que mantienen su forma a través de diferentes umbrales de intensidad. Es muy robusto a 
        cambios de iluminación y contraste, lo que lo hace ideal para detectar el interior sólido 
        de los paneles informativos, incluso con niebla.
        
        Referencia:
        - OpenCV MSER: https://docs.opencv.org/4.x/d3/d28/classcv_1_1MSER.html
        - Wikipedia MSER: https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions
        """
        engine = self.mser_fog if fog else self.mser
        regions, _ = engine.detectRegions(gray)
        return [cv2.boundingRect(r) for r in regions]

    # Postprocesa las detecciones
    def _postprocess(self, detections, h_img, w_img):
        detections = cluster_by_proximity(detections)
        detections = apply_nms(detections, self.nms_iou_threshold)
        detections = suppress_contained(detections, self.containment_threshold)
        
        margin = 5
        res = []
        for d in detections:
            x1, y1, x2, y2, sc = d
            cy = (y1 + y2) / 2
            if cy > h_img * self.bottom_roi_fraction: continue
            if cy > h_img * 0.93 and sc < 0.88: continue
            if y1 > h_img*0.5 and (y2-y1) < h_img*0.08 and sc < 0.82: continue
            if sum((x1 <= margin, x2 >= w_img - margin, y1 <= margin, y2 >= h_img - margin)) >= 2: continue
            if (x1<=margin or x2>=w_img-margin or y1<=margin or y2>=h_img-margin) and sc < self.edge_score_min: continue
            res.append(d)
        return res