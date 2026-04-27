import numpy as np

class Detector:
    """
    Clase base abstracta para detectores de paneles informativos.
    """
    def __init__(self):
        pass

    def train(self, train_path: str):
        """Prepara datos de entrenamiento si es necesario."""
        pass

    def detect(self, image: np.ndarray) -> list:
        """
        Detecta paneles informativos en una imagen.
        Returns: Lista de tuplas (x1, y1, x2, y2, score).
        """
        raise NotImplementedError("Las subclases deben implementar detect()")
