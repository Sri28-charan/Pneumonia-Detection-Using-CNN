
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



   
