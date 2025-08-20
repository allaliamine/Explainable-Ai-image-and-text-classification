
# Explainable ai for image and text classification

This project explores the integration of Explainable Artificial Intelligence (XAI) into deep learning models for both computer vision and natural language processing tasks. While deep neural networks achieve strong performance, their black-box nature makes it difficult to understand why certain predictions are made. By applying XAI techniques, this project aims to make model decisions more transparent, interpretable, and trustworthy especially in domains where reliability is crucial.



## Project Description

We trained deep learning models for two tasks:

- Image Classification (Medical Imaging) using the Bangladesh Brain Cancer MRI Dataset (6,056 images, 3 classes)

- Sentiment Analysis (Twitter Data) using the Sentiment140 dataset.

We then applied Explainable Ai techniques for image and text interpretability.

This repository contains code, datasets references, and visual examples demonstrating how XAI can uncover decision-making logic in deep learning models.

## Dataset

🧠 Medical Imaging: Bangladesh Brain Cancer MRI Dataset – 6,056 MRI images, 3 brain cancer types.

💬 Sentiment Analysis: Sentiment140 Dataset (Kaggle) – 1.6M tweets labeled positive/negative.
## Features

- Training deep learning models for image and text classification.
- Preprocessing pipeline for noisy social media text.
- Transfer learning with ResNet50 and MobileNetV2 for MRI images.
- Word embeddings with Word2Vec for NLP tasks.
- Model explainability with Grad-CAM, LIME, and SHAP.


## Installation & Usage

1. Clone the repository :

```bash
git clone https://github.com/allaliamine/Explainable-Ai-image-and-text-classification

cd Explainable-Ai-image-and-text-classification
```

2. Install requirements :
```bash
  pip install -r requirements.txt
```
3. Run both notebooks in the [Notebooks folder](https://github.com/allaliamine/Explainable-Ai-image-and-text-classification/tree/main/Notebooks).

> [!NOTE]  
> You can download the models, tokenizers, and explainers directly from this [Drive](https://drive.google.com/drive/folders/17tx6uP29TCEGWpqOTSo6tvb81UdRdPaD?usp=sharing) and place them in the Model folder.

4. Run the web interface : 
```bash
  flask run 
```

5. Choose the task to perform (image or text classification).  

6. Choose the explainability method :
- For **text classification**: use **SHAP** or **LIME**.  
- For **image classification**: use **Grad-CAM** or **LIME**.  


    
## Results

***1. Image Models :***
- MobileNetV2: 92.21% accuracy (best)
- Custom CNN: 76.97% accuracy
- ResNet50: 51.97% accuracy

***2. Text Models***

- Bi-RNN: 80.43% accuracy (best)
- Bi-LSTM: 79.81% accuracy
- Bi-GRU: 79.76% accuracy

***3. XAI Insights***

- Grad-CAM: Highlighted spatial tumor regions in MRI scans.
- LIME (image): Scattered superpixel-based regions, less intuitive.
- SHAP (text): Provided stable word-level importance scores.
- LIME (text): Highlighted key words but less robust than SHAP.
- 
## Contact

For any inquiries or feedback, please contact:
- [Allali Mohamed Amin](https://www.linkedin.com/in/m-amin-allali/)
