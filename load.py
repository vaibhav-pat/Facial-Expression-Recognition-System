from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
class FacialExpressionModel(object):
    EMOTIONS_LIST = [
                    "Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise", "Contempt"  # Add 8th common AffectNet class for safety
                    ]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        # If the image has 3 channels (BGR), convert it to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:  # Check if image has 3 channels (BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
        
        # Resize image to 48x48 (if it's not already)
        img = cv2.resize(img, (48, 48))
        
        # Keep original 0-255 scale to match training (no rescale used during training)
        img = img.astype("float32")
        
        # Ensure the image has the correct shape for the model (48, 48, 1)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension (grayscale)
        
        # Add batch dimension (for prediction) so the final shape is (1, 48, 48, 1)
        img = np.expand_dims(img, axis=0)   # Add batch dimension (shape becomes (1, 48, 48, 1))
        
        # Predict the emotion using the model
        self.preds = self.loaded_model.predict(img)
        
        # Return label for highest prediction, robust to mismatched label lengths
        top_index = int(np.argmax(self.preds))
        num_model_classes = int(self.preds.shape[-1]) if hasattr(self, 'preds') else len(FacialExpressionModel.EMOTIONS_LIST)
        if num_model_classes != len(FacialExpressionModel.EMOTIONS_LIST):
            # Fallback generic labels if counts differ
            fallback_labels = [f"Class_{i}" for i in range(num_model_classes)]
            return fallback_labels[top_index]
        return FacialExpressionModel.EMOTIONS_LIST[top_index]
