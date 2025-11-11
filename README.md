# 🧪 Skin Disease Classifier

A deep learning model to classify skin disease types from dermatoscopic images using CNNs in TensorFlow.

## 📊 Dataset

- Source: [Kaggle HAM10000 Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- ~10,000 dermatoscopic images
- 7 diagnosis classes (`nv`, `mel`, `bkl`, etc.)

## 🧠 Objective

To train a convolutional neural network (CNN) to accurately classify skin lesions into their respective categories for potential use in medical screening support.

## 🧪 Approach

1. **Data preprocessing:**  
   • Stratified 80/20 train-test split  
   • Resize to 224×224, normalize, cache for speed  
2. **Baseline CNNs:**  
   • Simple 2-layer model → ~68% val accuracy  
   • Deeper 3-layer model → ~67.5% val accuracy  
3. **Imbalance fixes:**  
   • Stratified sampling & class weights  
4. **Transfer learning:**  
   • MobileNetV2 base → ~67.5% val accuracy  
   • Fine-tuned (30 ep) → ~77.5% val accuracy  
   • Further fine-tuned → ~79.63% val accuracy  

---

## 📈 Results

| Model                              | Best Val Accuracy |
|------------------------------------|-------------------|
| 1. Simple CNN                      | 68.00%            |
| 2. Deeper CNN                      | 67.50%            |
| 3. Simple CNN + Stratified Sample  | 58.00%            |
| 4. Deeper CNN + Stratified Sample  | 67.50%            |
| 5. MobileNetV2 Transfer Learning   | 67.50%            |
| 6. MobileNetV2 + Fine-Tuning       | 77.50%            |
| 7. MobileNetV2 + Further Fine-Tune | 79.63%            |

*(See notebook for detailed graphs, confusion matrices, and training curves.)*

---

## 🧠 What I Learned

- Handling **real-world image data**: loading, resizing, caching  
- Tackling **class imbalance** with stratification and class weights  
- When **training from scratch** hits a ceiling on small datasets  
- How **transfer learning** and **fine-tuning** boost performance  

---

## 🚀 Future Work

- Experiment with stronger backbones (e.g., EfficientNet)  
- Add advanced augmentation (MixUp, CutMix)  
- Ensemble multiple models for higher accuracy  
- Deploy as a web demo or mobile app  

---

## ▶️ How to Run

```bash
git clone https://github.com/Allaawii/Skin-Disease-Classifier.git
cd Skin-Disease-Classifier
# Optional: set up a virtual environment with TensorFlow installed
jupyter notebook Skin_Disease_Classifier.ipynb
```
1. Open the notebook in Colab or locally.

2. Run cells in order: data prep → model training → evaluation.

3. Review the results, performance table, and conclusions.

## 📚 References

- [TensorFlow Deep Learning Course](https://github.com/mrdbourke/tensorflow-deep-learning)
- [HAM10000 Paper](https://arxiv.org/abs/1803.10417)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
