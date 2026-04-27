from models.interface import Detector
from models.mser_detector import DetectorBase
from models.hough_detector import DetectorHough

# Exportamos las clases para mantener compatibilidad con main.py
__all__ = ['Detector', 'DetectorBase', 'DetectorHough']