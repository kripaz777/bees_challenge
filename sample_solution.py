import sys
import os
import glob
import cv2
from ultralytics import YOLO

def get_latest_predict_folder(base_folder="runs/detect"):
    predict_folders = glob.glob(os.path.join(base_folder, "predict*"))
    if not predict_folders:
        raise FileNotFoundError("No predict folders found.")
    return max(predict_folders, key=os.path.getmtime)

def main():
    if len(sys.argv) != 2:
        print("Usage: python sample_solution.py <image_file>")
        sys.exit(1)

    image_file = sys.argv[1]
    model_path = "runs/detect/train/weights/best.pt"

    if not os.path.exists(image_file):
        print(f"❌ Image file '{image_file}' not found.")
        sys.exit(1)

    try:
        # Load trained model
        model = YOLO(model_path)

        # Predict → folder will be automatically incremented by YOLO
        results = model.predict(source=image_file, conf=0.25, save=True, show=False)

        # Count bees
        bee_count = len(results[0].boxes)
        print(f"Predicted Bee Count in '{image_file}': {bee_count}")

        # Find latest predict folder (handles predict, predict1, predict2, etc.)
        latest_pred_folder = get_latest_predict_folder()

        # Find predicted file in latest folder
        pred_file = os.path.join(latest_pred_folder, os.path.basename(image_file))
        if not os.path.exists(pred_file):
            print(f"❌ Predicted image not found: {pred_file}")
            sys.exit(1)

        # Read predicted image and overlay bee count
        img = cv2.imread(pred_file)
        text = f"Bee Count: {bee_count}"

        # Write text on the image (top-left corner)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Save final output image with bee count written
        output_file = os.path.join(latest_pred_folder, f"counted_{os.path.basename(image_file)}")
        cv2.imwrite(output_file, img)

        print(f"✅ Output image saved as: {output_file}")

        # Display the result window (optional)
        cv2.imshow("Bee Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
