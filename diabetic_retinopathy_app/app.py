import os
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf1
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import joblib

# Disable TF2 behavior
tf1.disable_v2_behavior()

app = Flask(__name__)

# --- Load Multi-class Classification Model (Frozen Graph) ---
multiclass_graph = tf1.Graph()
with multiclass_graph.as_default():
    graph_def = tf1.GraphDef()
    with open('model/output_graph.pb', "rb") as f:
        graph_def.ParseFromString(f.read())
        tf1.import_graph_def(graph_def, name='')
    sess_multi = tf1.Session(graph=multiclass_graph)

    # TEMPORARY: Print all operation names for debugging
    print("\n--- Available Operations in Frozen Graph ---")
    for op in multiclass_graph.get_operations():
        print(op.name)
    print("--- End of Operation List ---\n")

# ❗ Replace these once you see correct names in the printed list
input_tensor = multiclass_graph.get_tensor_by_name('Mul:0')  # Adjust this after inspecting printed ops
output_tensor = multiclass_graph.get_tensor_by_name('final_result:0')  # Adjust this too if needed

# --- Load Labels ---
with open('model/output_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# --- Load Binary Model and Scaler ---
keras_graph = tf1.Graph()
keras_sess = tf1.Session(graph=keras_graph)
with keras_graph.as_default():
    with keras_sess.as_default():
        binary_model = load_model('model/dr_ann_model.h5', compile=False)
scaler = joblib.load('model/binary_scaler.pkl')

# --- Preprocessing Functions ---
def preprocess_for_multiclass(image_path, target_size=(299, 299)):
    img = Image.open(image_path).convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

def preprocess_for_binary(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('L').resize(target_size)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    flat_img = img_array.flatten().reshape(1, -1)
    return scaler.transform(flat_img)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img_path = os.path.join('static', 'uploads', file.filename)
        file.save(img_path)

        # --- Binary Classification ---
        with keras_graph.as_default():
            with keras_sess.as_default():
                binary_input = preprocess_for_binary(img_path)
                binary_prediction = binary_model.predict(binary_input)[0][0]
                dr_present = binary_prediction >= 0.5

        # --- Multi-class Classification ---
        image = preprocess_for_multiclass(img_path)
        with multiclass_graph.as_default():
            predictions = sess_multi.run(output_tensor, {input_tensor: [image]})
        stage_results = {labels[i]: float(predictions[0][i]) for i in range(len(labels))}
        top_stage_prediction = max(stage_results, key=stage_results.get)

        return jsonify({
            'image_path': img_path,
            'dr_present': bool(dr_present),
            'binary_confidence': float(binary_prediction),
            'dr_stage': top_stage_prediction,
            'stage_confidence': float(stage_results[top_stage_prediction]),
            'all_stage_predictions': stage_results
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
