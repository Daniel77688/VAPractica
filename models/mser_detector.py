import cv2
import numpy as np
from .interface import Detector
from utils.vision import (
    apply_nms, suppress_contained,
    estimate_fog_conditions, build_ideal_mask, enhance_image_clahe
)

class DetectorBase(Detector):
    def __init__(self):
        super().__init__()
        # Configuración MSER
        self.mser = cv2.MSER_create(delta=3, min_area=600, max_area=100000, max_variation=0.40, min_diversity=0.2) # Algoritmo MSER para regiones estables

        # Rangos HSV
        self.hsv_lower = np.array([100, 130, 80]) # Límite inferior azul (Normal)
        self.hsv_upper = np.array([130, 255, 255]) # Límite superior azul (Normal)
        self.hsv_lower_fog = np.array([95, 45, 55]) # Límite inferior azul (Niebla)
        self.hsv_upper_fog = np.array([135, 255, 255]) # Límite superior azul (Niebla)


        # Parámetros y Umbrales
        self.min_blue_ratio = 0.50 # Ratio mínimo de azul en ROI (Normal)
        self.min_blue_ratio_fog = 0.22 # Ratio mínimo de azul en ROI (Niebla)
        self.min_blue_ratio_fog_fallback = 0.16 # Ratio de azul para pase de rescate
        
        self.min_aspect_ratio = 0.5 # Relación ancho/alto mínima
        self.max_aspect_ratio = 5.0 # Relación ancho/alto máxima
        self.min_bbox_area = 800 # Área mínima del cuadro delimitador
        self.max_bbox_area = 200000 # Área máxima del cuadro delimitador
        self.min_dimension = 30 # Tamaño mínimo en píxeles (w y h)
        
        self.score_threshold = 0.58 # Umbral de score F1 (Normal)
        self.score_threshold_fog = 0.70 # Umbral de score F1 (Niebla)

        self.score_threshold_fog_fallback = 0.48 # Umbral para rescate en niebla
        self.high_score_override = 0.92 # Score que ignora validación de bordes
        
        self.color_expand = 0.02 # Margen extra para detecciones de color
        self.mser_expand = 0.02 # Margen extra para detecciones MSER
        
        # Post-procesamiento
        self.nms_iou_threshold = 0.45 # Umbral de solapamiento para NMS
        self.containment_threshold = 0.65 # Umbral de contención entre cajas
        self.score_replacement_diff = 0.12 # Diferencia de score para absorción
        self.bottom_roi_fraction = 0.98 # Fracción de imagen útil (evita suelo)
        self.low_roi_y_ratio = 0.93 # Y partir del cual se sube exigencia
        self.low_roi_score_min = 0.88 # Score mínimo en zona inferior
        self.small_sign_h_ratio = 0.08 # Altura relativa para cartel "pequeño"
        self.small_sign_score_min = 0.82 # Score mínimo para carteles pequeños
        self.edge_score_min = 0.75 # Score mínimo si toca bordes de imagen
        self.edge_margin = 5 # Píxeles de margen para considerar borde
        
        # Borde claro (Validación de borde blanco alrededor)
        self.border_margin_ratio = 0.08 # Grosor del área de borde a mirar
        self.border_brightness_min = 90 # Brillo mínimo del borde
        self.border_saturation_max = 140 # Saturación máxima del borde
        self.min_bright_borders = 1 # Mínimo de lados claros requeridos
        
        # Heurística niebla
        self.fog_sat_mean_max = 27.0 # Saturación media máxima para niebla
        self.fog_edge_ratio_max = 0.015 # Ratio de bordes máximo para niebla

        # Máscara ideal
        self.mask_h, self.mask_w = 40, 80 # Tamaño de redimensionado para F1
        self.ideal_mask = build_ideal_mask(self.mask_h, self.mask_w) # Plantilla ideal
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # Kernel morfológico normal
        self.morph_kernel_fog = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # Kernel morfológico niebla

    @staticmethod
    def _is_valid_geometry(w, h, min_aspect_ratio, max_aspect_ratio, min_bbox_area, max_bbox_area, min_dimension):
        if h <= 0: return False
        aspect = w / h
        area = w * h
        return (min_aspect_ratio <= aspect <= max_aspect_ratio and min_bbox_area <= area <= max_bbox_area and w >= min_dimension and h >= min_dimension)

    # Método principal de detección
    def detect(self, image: np.ndarray) -> list:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_eq = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        is_fog = estimate_fog_conditions(hsv_image, gray_eq, self.fog_sat_mean_max, self.fog_edge_ratio_max)
        
        # Si detectamos niebla, mejoramos la imagen con CLAHE antes de empezar
        if is_fog:
            image, hsv_image, gray_eq = enhance_image_clahe(image)

        h_img, w_img = image.shape[:2]

        # Pases de detección
        cfg_normal = self._create_cfg(self.hsv_lower, self.hsv_upper, self.min_blue_ratio, self.score_threshold)
        cfg_fog = self._create_cfg(self.hsv_lower_fog, self.hsv_upper_fog, self.min_blue_ratio_fog, self.score_threshold_fog, small=True, fog_mser=True)

        passes = [cfg_normal]
        if is_fog: passes.append(cfg_fog)
        
        candidates = self._run_pass(image, hsv_image, gray_eq, passes)
        final = apply_nms(candidates, self.nms_iou_threshold)

        # Pase de rescate si no se ha detectado nada bajo la niebla
        if is_fog and not final:
            cfg_fallback = self._create_cfg(self.hsv_lower_fog, self.hsv_upper_fog, self.min_blue_ratio_fog_fallback, self.score_threshold_fog_fallback, small=True, fog_mser=True, border=False)
            final = apply_nms(self._run_pass(image, hsv_image, gray_eq, [cfg_fallback]), self.nms_iou_threshold)


        return self._postprocess(final, h_img, w_img)

    # Crea una configuración de detección
    def _create_cfg(self, lo, hi, ratio, thr, small=False, fog_mser=False, border=True):
        return {"hsv_lower": lo, "hsv_upper": hi, "min_blue_ratio": ratio, "score_threshold": thr, "small_kernel": small, "use_fog_mser": fog_mser, "require_border": border}

    # Ejecuta un pase de detección
    def _run_pass(self, image, hsv_image, gray_eq, pass_configs):
        candidates = []
        for cfg in pass_configs:
            color_rects = self._get_color_candidates(hsv_image, cfg["hsv_lower"], cfg["hsv_upper"], cfg.get("small_kernel", False))
            mser_rects = self._get_mser_candidates(gray_eq, cfg.get("use_fog_mser", False))

            for rect_list, expand in [(color_rects, self.color_expand), (mser_rects, self.mser_expand)]:
                for rect in rect_list:
                    res = self._evaluate_candidate(image, hsv_image, *rect, expand, cfg)
                    if res: candidates.append(res)
        return candidates


    # Evalúa un candidato
    def _evaluate_candidate(self, image, hsv_image, x, y, w, h, expand, cfg):
        h_img, w_img = image.shape[:2]
        if not self._is_valid_geometry(w, h, self.min_aspect_ratio, self.max_aspect_ratio, self.min_bbox_area, self.max_bbox_area, self.min_dimension): return None

        pad_x, pad_y = int(w * expand), int(h * expand)
        x1, y1, x2, y2 = max(0, x-pad_x), max(0, y-pad_y), min(w_img, x+w+pad_x), min(h_img, y+h+pad_y)
        
        roi_hsv = hsv_image[y1:y2, x1:x2]
        if roi_hsv.size == 0: return None
        
        if self._compute_blue_ratio(roi_hsv, cfg["hsv_lower"], cfg["hsv_upper"]) < cfg["min_blue_ratio"]: return None

        score = self._compute_f1_score(image[y1:y2, x1:x2], cfg["hsv_lower"], cfg["hsv_upper"])
        if score < cfg["score_threshold"]: return None

        if cfg.get("require_border", True):
            if not self._check_border(hsv_image, x1, y1, x2, y2): # Comprueba borde blanco alrededor
                if score < self.high_score_override: return None # Si tiene nota alta, es un panel

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
        box_w = max(1, x2 - x1)
        m = max(1, int(box_w * self.border_margin_ratio))
        strips = [(y1-m, y1, x1, x2, y1-m>=0), (y2, y2+m, x1, x2, y2+m<=h_img), (y1, y2, x1-m, x1, x1-m>=0), (y1, y2, x2, x2+m, x2+m<=w_img)] # Franjas de borde
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
        mask = cv2.inRange(hsv, lo, hi) # Máscara de color
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel_fog if small else self.morph_kernel) # Cierre morfológico
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Extracción de contornos
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
        engine = self.mser # Motor de detección MSER
        regions, _ = engine.detectRegions(gray) # Regiones detectadas
        return [cv2.boundingRect(r) for r in regions]

    # Postprocesa las detecciones
    def _postprocess(self, detections, h_img, w_img):
        detections = suppress_contained(detections, self.containment_threshold, self.score_replacement_diff)
        
        res = []
        for d in detections:
            x1, y1, x2, y2, sc = d
            cy = (y1 + y2) / 2
            if cy > h_img * self.bottom_roi_fraction: continue # Si está en la franja inferior, se descarta
            if cy > h_img * self.low_roi_y_ratio and sc < self.low_roi_score_min: continue # Si está en franja baja con baja confianza, se descarta
            if y1 > h_img*0.5 and (y2-y1) < h_img * self.small_sign_h_ratio and sc < self.small_sign_score_min: continue # Si es pequeño y está en mitad inferior con baja confianza, se descarta
            if sum((x1 <= self.edge_margin, x2 >= w_img - self.edge_margin, y1 <= self.edge_margin, y2 >= h_img - self.edge_margin)) >= 2: continue # Si está en dos lados, se descarta
            if (x1<=self.edge_margin or x2>=w_img-self.edge_margin or y1<=self.edge_margin or y2>=h_img-self.edge_margin) and sc < self.edge_score_min: continue # Si está en un lado con baja confianza, se descarta
            res.append(d)
        return res