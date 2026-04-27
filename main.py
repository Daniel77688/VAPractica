# main.py
# ============================================================================
# Script principal para la detección de paneles informativos de autopista.
#
# Uso:
#   python main.py --train_path ./train_detection --test_path ./test_detection --detector BASE
#
# Salidas:
#   - resultado_imgs/  → imágenes de test con bounding boxes y scores dibujados.
#   - resultado.txt    → fichero con todas las detecciones en formato evaluable.
#
# Asignatura: Visión Artificial – Universidad Rey Juan Carlos
# ============================================================================

import argparse
import os
import cv2
import numpy as np
from detector import Detector, DetectorBase


def load_images(path: str) -> list:
    """
    Carga todas las imágenes PNG de un directorio, ordenadas por nombre.

    Args:
        path: ruta al directorio con las imágenes.

    Returns:
        Lista de tuplas (nombre_fichero, imagen_BGR).
    """
    images = []
    for fname in sorted(os.listdir(path)):
        if fname.lower().endswith('.png'):
            img_path = os.path.join(path, fname)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((fname, img))
    return images


def create_detector(detector_name: str) -> Detector:
    """
    Factoría de detectores: crea la instancia adecuada según el nombre.

    Facilita la extensión con nuevos detectores sin modificar el resto del código.

    Args:
        detector_name: nombre del detector (p.ej. "BASE").

    Returns:
        Instancia de Detector configurada.

    Raises:
        ValueError: si el nombre del detector no es reconocido.
    """
    name = detector_name.upper()
    if name == 'BASE':
        return DetectorBase()
    if name == 'HOUGH':
        from detector import DetectorHough
        return DetectorHough()
    raise ValueError(f"Detector desconocido: '{detector_name}'. Opciones disponibles: BASE, HOUGH")


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Dibuja las detecciones sobre una copia de la imagen.

    Cada detección se visualiza con:
      - Rectángulo rojo (BGR: 0,0,255) alrededor del bounding box.
      - Texto amarillo (BGR: 0,255,255) con el score encima del rectángulo.

    Args:
        image: imagen BGR original.
        detections: lista de tuplas (x1, y1, x2, y2, score).

    Returns:
        Copia de la imagen con las detecciones dibujadas.
    """
    result = image.copy()
    for (x1, y1, x2, y2, score) in detections:
        # Rectángulo rojo
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Texto con el score en amarillo, encima del rectángulo
        label = f"{score:.2f}"
        text_y = max(y1 - 5, 15)  # Evitar que el texto se salga por arriba
        cv2.putText(result, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="BASE",
        help='Detector string name (default: BASE)')
    parser.add_argument(
        '--train_path', default="",
        help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="",
        help='Select the testing data dir')
    parser.add_argument(
        '--min_score', type=float, default=0.0,
        help='Confidence threshold to filter detections before saving (default: 0.0)')

    args = parser.parse_args()

    # Crear directorio de salida para las imágenes con detecciones
    output_img_dir = os.path.join(os.getcwd(), "resultado_imgs")
    os.makedirs(output_img_dir, exist_ok=True)

    # Crear el detector según el argumento recibido
    detector = create_detector(args.detector)
    print(f"Detector creado: {args.detector.upper()}")

    # Fase de entrenamiento (si el detector lo requiere)
    if args.train_path:
        print(f"Cargando datos de entrenamiento desde: {args.train_path}")
        detector.train(args.train_path)

    # Cargar imágenes de test
    if not args.test_path:
        print("ERROR: No se ha especificado --test_path")
        exit(1)

    print(f"Cargando imágenes de test desde: {args.test_path}")
    test_images = load_images(args.test_path)
    print(f"Imágenes de test cargadas: {len(test_images)}")

    # Procesar cada imagen: detectar, dibujar y guardar
    result_lines = []
    for idx, (fname, image) in enumerate(test_images):
        # Ejecutar detección
        detections = detector.detect(image)
        detections = [d for d in detections if d[4] >= args.min_score]

        # Dibujar detecciones sobre la imagen
        result_img = draw_detections(image, detections)

        # Guardar imagen con detecciones
        output_path = os.path.join(output_img_dir, fname)
        cv2.imwrite(output_path, result_img)

        # Registrar cada detección en formato evaluable
        for (x1, y1, x2, y2, score) in detections:
            result_lines.append(f"{fname};{x1};{y1};{x2};{y2};1;{score:.4f}")

        # Progreso
        if (idx + 1) % 10 == 0 or idx == len(test_images) - 1:
            print(f"  Procesadas {idx + 1}/{len(test_images)} imágenes...")

    # Guardar fichero de resultados
    with open("resultado.txt", 'w') as f:
        f.write('\n'.join(result_lines))
        if result_lines:
            f.write('\n')

    print(f"\n=== Resumen ===")
    print(f"Total detecciones: {len(result_lines)}")
    print(f"Imágenes procesadas: {len(test_images)}")
    print(f"Imágenes guardadas en: {output_img_dir}")
    print(f"Resultados guardados en: resultado.txt")
