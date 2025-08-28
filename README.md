
#  Pneumonia Detection with MobileNetV2

This project applies **Transfer Learning with MobileNetV2** to classify chest X-rays as **Normal** or **Pneumonia**. Implemented in **Google Colab** using the **Kaggle Chest X-Ray dataset**.

---

## 📖 Workflow

1. **Data Preparation**

   * Train/Test/Validation sets from Kaggle dataset
   * Preprocessing & augmentation (`ImageDataGenerator`)

2. **Model Architecture**

   * Base: **MobileNetV2 (ImageNet, include\_top=False)**
   * Custom layers: GAP → Dropout → Dense(128, ReLU) → Dropout → Dense(1, Sigmoid)
   * Loss: Binary Crossentropy | Optimizer: Adam (lr=1e-4)

3. **Training**

   * EarlyStopping (patience=5) & ModelCheckpoint
   * Trained for max 25 epochs (stopped early if no improvement)
   * Best model saved as `mobilenetv2_pneumonia.h5`

4. **Fine-Tuning**

   * Unfroze last 20 layers of MobileNetV2
   * Recompiled with Adam (lr=1e-5), metrics: Accuracy, AUC, Precision, Recall
   * Trained for 30 epochs, best model saved as `mobilenetv2_pneumonia_finetuned.h5`

5. **Inference**

   * Load saved model
   * Preprocess single X-ray (150×150, RGB, normalized)
   * Predict probability → classify as **Normal** or **Pneumonia**

---

## 📊 Results

  - Accuracy and Loss before Finetuning
     . Test Accuracy: 86.06%
     . Test Loss: 0.3284

  - Results After Finetuning
     . Accuracy: 85.84%  
     . AUC: 94.38%  
     . Loss: 35.03%  
     . Precision: 83.86%  
     . Recall: 96.12%
## Output 
 - The Model Detect the penumonia with 99.94% Confidence
    
## 📂 Dataset  

Chest X-Ray Images (Pneumonia) dataset:  
👉 [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


### 📌 Run in Google Colab  
Click below to open the notebook in Colab:  


###Open it directly in **Google Colab**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Snl28-charan/Stock-Price-Prediction/blob/main/Stock_Price_Prediction.ipynb)


   
