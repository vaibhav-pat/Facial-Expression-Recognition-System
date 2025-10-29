## Facial Expression Recognition (FRS)

Real-time facial expression recognition using a CNN (48x48 grayscale) with 8 classes. Provides webcam inference and an optional training script.

### Environment

```bash
cd /Users/vaibhavpatidar/Downloads/FRS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, ensure the terminal/IDE has camera permissions (System Settings → Privacy & Security → Camera).

### Run Inference (Webcam)

```bash
python prediction.py --camera 0 --title "Facial Expression Recognition" --quit q
```

Arguments:
- `--camera`: camera index (default 0)
- `--title`: window title
- `--quit`: quit key (default 'q')

### Train the Model (optional)

Prepare a dataset directory with the structure:

```
DATASET_DIR/
  train/
    Class_1/ ... images ...
    Class_2/ ...
    ...
  test/
    Class_1/ ... images ...
    Class_2/ ...
    ...
```

Run training:

```bash
python main.py \
  --dataset_dir "/absolute/path/to/DATASET_DIR" \
  --epochs 20 \
  --batch_size 64 \
  --img_size 48 \
  --output_json model.json \
  --output_weights model_weights.h5
```

Outputs:
- `model.json`: Keras model architecture
- `model_weights.h5`: trained weights

### Notes
- The training script uses only built-in Keras callbacks and no plotting libs.
- Keep Keras/TensorFlow versions consistent with `requirements.txt` (2.13.x) to avoid weight loading issues.


# Facial-Expression-Recognition-System
