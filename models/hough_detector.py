import cv2
import numpy as np
from .mser_detector import DetectorBase
from utils.vision import apply_nms

# Clase detector alternativo basado en Transformada de Hough.
class DetectorHough(DetectorBase):

    def __init__(self):
        super().__init__()

    def detect(self, image: np.ndarray) -> list:
        """
        Detecta paneles informativos utilizando la Transformada de Hough probabilística.
        
        Motivo de la estrategia: En lugar de buscar rectángulos directamente en bordes caóticos, 
        primero filtramos la imagen por el color del panel (azul). Luego, usamos HoughLinesP para 
        detectar segmentos rectos en esos bordes. Finalmente, aplicamos un cierre morfológico 
        (Morphological Close) para fusionar líneas cercanas y formar bloques sólidos que representan 
        el panel completo.
        
        Referencias:
        - Probabilistic Hough Transform: https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
        - Operador bitwise_and (Enmascaramiento): https://docs.opencv.org/3.4/d0/d86/tutorial_py_image_arithmetics.html
        - Cierre Morfológico (MORPH_CLOSE): https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        """
        h_img, w_img = image.shape[:2]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Crear máscara de color azul para reducir ruido de fondo
        blue_mask = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)
        
        # Dilatar la máscara un poco para asegurar que incluya los bordes exteriores del panel
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blue_mask_dilated = cv2.dilate(blue_mask, kernel_dilate)

        # 2. Obtener bordes de la imagen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 3. Filtrar bordes: solo mantener los que están en regiones azules
        edges_filtered = cv2.bitwise_and(edges, edges, mask=blue_mask_dilated)

        # 4. Encontrar líneas de Hough
        lines = cv2.HoughLinesP(edges_filtered, rho=1, theta=np.pi/180, threshold=25, minLineLength=15, maxLineGap=20)
        candidates = []

        if lines is not None:
            # 5. Dibujar líneas encontradas en una máscara con grosor amplio
            hough_mask = np.zeros_like(edges)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 4)

            # 6. Operaciones morfológicas para conectar las líneas fragmentadas
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            hough_closed = cv2.morphologyEx(hough_mask, cv2.MORPH_CLOSE, close_kernel)
            
            # Encontrar los contornos de las agrupaciones de líneas
            contours, _ = cv2.findContours(hough_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                
                # Expandir un poco el bounding box para cubrir todo el panel real
                pad_x, pad_y = int(w * 0.05), int(h * 0.05)
                x1, y1, x2, y2 = max(0, x-pad_x), max(0, y-pad_y), min(w_img, x+w+pad_x), min(h_img, y+h+pad_y)
                w_pad = x2 - x1
                h_pad = y2 - y1

                # Validaciones geométricas
                if not self._is_valid_geometry(
                    w_pad, h_pad,
                    self.min_aspect_ratio,
                    self.max_aspect_ratio,
                    self.min_bbox_area,
                    self.max_bbox_area,
                    self.min_dimension,
                ):
                    continue

                roi_hsv = hsv_image[y1:y2, x1:x2]
                if roi_hsv.size == 0: continue
                
                # Validar la proporción de azul usando el ratio configurado
                if self._compute_blue_ratio(roi_hsv, self.hsv_lower, self.hsv_upper) < (self.min_blue_ratio * 0.8):
                    continue

                score = self._compute_f1_score(image[y1:y2, x1:x2], self.hsv_lower, self.hsv_upper)
                if score < self.score_threshold: 
                    continue

                candidates.append((x1, y1, x2, y2, float(score)))

        detections = apply_nms(candidates, self.nms_iou_threshold)
        return self._postprocess(detections, h_img, w_img)

