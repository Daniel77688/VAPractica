import argparse, cv2
from pathlib import Path
from detector import DetectorBase, DetectorHough

def main():
    parser = argparse.ArgumentParser(description='Detección de paneles informativos')
    parser.add_argument('--detector', default="BASE", help='BASE o HOUGH')
    parser.add_argument('--train_path', default="")
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--min_score', type=float, default=0.0)
    args = parser.parse_args()

    # 1. Inicializar detector
    detector = DetectorHough() if args.detector.upper() == 'HOUGH' else DetectorBase()
    if args.train_path:
        detector.train(args.train_path)

    # 2. Preparar salida
    out_dir = Path("resultado_imgs")
    out_dir.mkdir(exist_ok=True)
    results = []

    # 3. Procesamiento
    print(f"Usando detector: {args.detector.upper()}")
    for img_path in sorted(Path(args.test_path).glob("*.png")):
        img = cv2.imread(str(img_path))
        detections = [d for d in detector.detect(img) if d[4] >= args.min_score]

        for x1, y1, x2, y2, score in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{score:.2f}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            results.append(f"{img_path.name};{x1};{y1};{x2};{y2};1;{score:.4f}")

        cv2.imwrite(str(out_dir / img_path.name), img)

    # 4. Guardar resultados
    Path("resultado.txt").write_text("\n".join(results) + "\n")
    print(f"Finalizado: {len(results)} detecciones en {len(list(Path(args.test_path).glob('*.png')))} imágenes.")

if __name__ == "__main__":
    main()
