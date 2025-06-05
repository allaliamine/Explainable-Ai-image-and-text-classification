from flask import Flask, render_template, request, redirect, url_for
from Functions.cleaners import *
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
import pickle
import tensorflow as tf
import shap
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def predict_proba(texts):
    
    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=81, padding='post')
    probs = sentiment_model.predict(padded_seqs)
    probs_all = np.hstack([1 - probs, probs])
    
    return probs_all

app = Flask(__name__)

def init():
    global sentiment_model, tokenizer, brain_tumor_model, explainer

    # Load the models
    sentiment_model = tf.keras.models.load_model('/Users/mac/Desktop/Explainable-Ai-image-and-text-classification/Model/Sentiment_Analysis_model/sentiment_analysis_model.h5')

    # Load the tokenizer
    with open('/Users/mac/Desktop/Explainable-Ai-image-and-text-classification/Model/Sentiment_Analysis_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the brain tumor model
    brain_tumor_model = tf.keras.models.load_model('/Users/mac/Desktop/Explainable-Ai-image-and-text-classification/Model/Brain_MRI_model/Brain_Tumor_model.h5')

    # Load the SHAP explainer
    with open("/Users/mac/Desktop/Explainable-Ai-image-and-text-classification/Model/ShapExplainer/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)


def clean_text(text):

    clean_text = perform_all_cleaning(text)
    return clean_text

@app.route('/')
def index():
    init()
    prediction = request.args.get('prediction', '')
    proba = request.args.get('proba', 0)
    text = request.args.get('text', '')
    
    return render_template('index.html', prediction=prediction, proba=proba, text=text, method='')



@app.route('/explain/lime', methods=['POST'])
def predict():

    text = request.form['text']
    # text = clean_text(text)

    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=81, padding='post')

    probas = sentiment_model.predict(padded)[0]
    positive_prob = probas[0]
    predicted_label = 'Positive' if positive_prob > 0.5 else 'Negative'
    confidence = round(100 * positive_prob, 2) if predicted_label == 'Positive' else round(100 * (1 - positive_prob), 2)

    method = "LIME"
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    explanation_html = exp.as_html()

    return render_template('index.html', explanation_html=explanation_html, text=text, prediction=predicted_label, proba=confidence, method=method)



@app.route('/explain/shap', methods=['POST'])
def explain_shap():

    text = request.form['text']
    text_clean = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=81, padding='post')

    shap_values = explainer(padded)

    index_word = {v: k for k, v in tokenizer.word_index.items()}
    words = [index_word.get(idx, '[PAD]' if idx == 0 else '[UNK]') for idx in padded[0]]

    base_val = shap_values[0].base_values
    if not np.isscalar(base_val):
        base_val = base_val[0]

    filtered_words = []
    filtered_values = []
    for w, v in zip(words, shap_values[0].values):
        if w != '[PAD]':
            filtered_words.append(w)
            filtered_values.append(v)

    filtered_words_spaced = [w + ' ' for w in filtered_words]

    raw_words = text.split()  

    display_words = []
    raw_idx = 0
    for idx in padded[0]:
        if idx == 0:
            continue  
        if raw_idx < len(raw_words):
            display_words.append(raw_words[raw_idx] + ' ')
            raw_idx += 1
        else:
            display_words.append('[UNK] ')

    fixed_shap_values = shap.Explanation(
        values=np.array(filtered_values),
        base_values=base_val,
        data=filtered_words_spaced,
        feature_names=display_words,
        display_data=display_words
    )

    shap_html = shap.plots.text(fixed_shap_values, display=False)

    probas = sentiment_model.predict(padded)[0]
    positive_prob = probas[0]
    predicted_label = 'Positive' if positive_prob > 0.5 else 'Negative'
    confidence = round(100 * positive_prob, 2) if predicted_label == 'Positive' else round(100 * (1 - positive_prob), 2)

    method = "SHAP"

    return render_template('index.html', shap_html=shap_html, text=text, prediction=predicted_label, proba=confidence, method=method)




@app.route('/explain/gradcam', methods=['POST'])
def explain_gradcam():
    image = request.files['image']
    if not image:
        return redirect(url_for('index'))

    img_path = os.path.join('static', 'uploaded_img.jpg')
    image.save(img_path)

    # Preprocess image
    img_pil = Image.open(img_path).resize((224, 224))
    img_array = np.array(img_pil) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    class_names = ['brain_glioma', 'brain_menin', 'brain_tumor']

    grad_model = tf.keras.models.Model(
        [brain_tumor_model.inputs],
        [brain_tumor_model.get_layer('out_relu').output, brain_tumor_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224))
    heatmap = np.array(heatmap_img) / 255.0
    heatmap_colormap = cm.jet(heatmap)[:, :, :3]

    superimposed_img = heatmap_colormap * 0.4 + np.array(img_pil) / 255.0
    superimposed_img = np.clip(superimposed_img, 0, 1)

    # Save Grad-CAM result
    gradcam_path = os.path.join('static', 'gradcam_result.jpg')
    plt.imsave(gradcam_path, superimposed_img)

    predicted_label = class_names[int(class_index)]
    confidence = tf.nn.softmax(predictions[0])[class_index].numpy()
    method = "Grad-CAM"

    return render_template(
        'index.html',
        prediction=predicted_label,
        proba=round(100 * confidence, 2),
        method=method,
        gradcam_image=gradcam_path
    )


@app.route('/explain/lime_image', methods=['POST'])
def explain_lime():
    image = request.files.get('image')
    if not image:
        return redirect(url_for('index'))

    img_path = os.path.join('static', 'uploaded_img.jpg')
    image.save(img_path)

    img_pil = Image.open(img_path)
    img_array = np.array(img_pil) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    class_names = ['brain_glioma', 'brain_menin', 'brain_tumor']

    predictions = brain_tumor_model.predict(img_array)
    class_index = tf.argmax(predictions[0]).numpy()

    explainer = LimeImageExplainer()

    def predict_fn(images):
        images = np.array(images)
        images = images / 255.0 if images.max() > 1 else images
        return brain_tumor_model.predict(images)

    explanation = explainer.explain_instance(
        np.array(img_pil),
        classifier_fn=predict_fn,
        top_labels=3,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=class_index,
        positive_only=True,
        hide_rest=False,
        num_features=5,
        min_weight=0.0
    )

    img_with_boundaries = mark_boundaries(np.array(img_pil) / 255.0, mask)

    lime_path = os.path.join('static', 'lime_result.jpg')
    plt.imsave(lime_path, img_with_boundaries)

    predicted_label = class_names[class_index]
    confidence = tf.nn.softmax(predictions[0])[class_index].numpy()
    method = "LIME_image"

    return render_template(
        'index.html',
        prediction=predicted_label,
        proba=round(100 * confidence, 2),
        method=method,
        lime_image=lime_path
    )


if __name__ == '__main__':
    app.run(debug=True)
