# 🔬 Kaggle LLM Science Exam: QLoRA-Enhanced Scientific QA System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A3%97%20Hugging%20Face-Canl%C4%B1%20Demo-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/sancos/llm-science-model)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Yar%C4%B1%C5%9Fma-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

# 🏗️ 1. PROJE ÖZETİ 

## 1.1 Genel Bakış ve Amaç

Bu proje, yüksek düzeyde akademik ve teknik uzmanlık gerektiren çoktan seçmeli bilimsel soruların Büyük Dil Modelleri (LLM) yardımıyla çözülmesini amaçlayan gelişmiş bir üretken yapay zeka sistemidir.

Çalışma kapsamında aşağıdaki disiplinler hedeflenmiştir:

- Fizik
- Kimya
- Biyoloji
- Bilgisayar Bilimleri
- Kozmoloji
- Astronomi
- Genel Bilimsel Muhakeme Problemleri

Proje altyapısı, Kaggle platformunda yayınlanan **Kaggle - LLM Science Exam** yarışma veri seti üzerine inşa edilmiştir.

---

## 1.2 Mühendislik Yaklaşımı ve Sistem Dönüşümü

Projenin ilk fazlarında klasik encoder tabanlı transformer mimarileri değerlendirilmiştir. Özellikle:

- `DeBERTa-v3-base`
- `RoBERTa-base`
- `DistilBERT`

modelleri test edilmiştir.

Ancak veri setinin düşük hacimli olması (~200 satır) ve soruların ileri düzey akademik terminoloji içermesi nedeniyle klasik encoder mimarileri yeterli semantik muhakeme performansı sağlayamamıştır.

Bu nedenle sistem mimarisi tamamen yeniden tasarlanmış ve modern üretken yapay zeka yaklaşımına geçiş yapılmıştır.

Yeni sistemde aşağıdaki teknoloji yığını kullanılmıştır:

| Bileşen | Kullanılan Teknoloji |
|---|---|
| Ana Model | Qwen2.5-1.5B-Instruct |
| Fine-Tuning | QLoRA |
| Kuantizasyon | 4-Bit NF4 |
| Eğitim Altyapısı | Hugging Face Transformers |
| GPU Platformu | Kaggle T4 GPU |
| Adaptör Eğitimi | PEFT / LoRA |

---

## 1.3 Temel Başarı Metrikleri

| Metrik | Sonuç |
|---|---|
| Başlangıç Accuracy | %20.00 |
| Nihai Accuracy | **%80.00** |
| Training Loss | 0.9124 |
| Validation Loss | 1.0829 |
| GPU | NVIDIA T4 (16GB VRAM) |

---

# 🧬 2. SİSTEM MİMARİSİ VE METODOLOJİ

## 2.1 Choice-Merged Prompt Engineering Yaklaşımı

Klasik multiple-choice sistemlerde her seçenek ayrı tokenize edilir. Ancak bu yöntem:

- token maliyetini artırır,
- semantik bağlamı parçalar,
- reasoning kalitesini düşürür.

Bu projede modern LLM mimarilerine daha uygun olan **Choice-Merged Prompt Formatı** kullanılmıştır.

Örnek prompt yapısı:

```text
Answer the following multiple choice question.

Question:
What causes the expansion of the universe?

Choices:
A. Dark matter
B. Quantum tunneling
C. Gravity waves
D. Magnetic collapse
E. Dark energy

Answer:
```

Bu yöntem sayesinde model:

- tüm seçenekleri aynı bağlam içinde değerlendirir,
- semantik kıyaslama yapabilir,
- reasoning kapasitesini daha verimli kullanır.

---

## 2.2 QLoRA ve Kuantizasyon Parametreleri

Sistemin düşük VRAM ortamlarında çalışabilmesi için modern QLoRA yaklaşımı tercih edilmiştir.

### Kullanılan Parametreler

| Parametre | Değer |
|---|---|
| Quantization | 4-Bit NF4 |
| Double Quantization | True |
| LoRA Rank (r) | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| Optimizer | paged_adamw_8bit |
| Precision | FP16 |
| Max Sequence Length | 512 |

---

## 2.3 Hedeflenen Attention Katmanları

LoRA adaptörleri yalnızca attention mekanizmasının kritik bileşenlerine uygulanmıştır:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

Bu yaklaşım sayesinde:

- eğitim maliyeti düşürülmüş,
- VRAM tüketimi azaltılmış,
- inference hızı korunmuştur.

---

## 2.4 Bellek Yönetimi Stratejileri

Kaggle T4 GPU üzerinde stabil eğitim için aşağıdaki optimizasyonlar uygulanmıştır:

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Cache Devre Dışı Bırakma

```python
model.config.use_cache = False
```

### 4-Bit Quantization

```python
bnb_4bit_quant_type = "nf4"
```

Bu teknikler sayesinde:

- CUDA OOM hataları önlenmiş,
- GPU stabilitesi korunmuş,
- eğitim süreci kesintisiz tamamlanmıştır.

---

# 📊 3. PERFORMANS ANALİZİ

## 3.1 Classification Report

Validation set üzerinde elde edilen nihai performans sonuçları:

```text
              precision    recall  f1-score   support

           A       0.80      1.00      0.89         4
           B       0.80      0.80      0.80         5
           C       0.75      0.75      0.75         4
           D       0.80      1.00      0.89         4
           E       1.00      0.33      0.50         3

    accuracy                           0.80        20
   macro avg       0.83      0.78      0.77        20
weighted avg       0.82      0.80      0.78        20
```

---

## 3.2 Canlı Tahmin Analizleri

### ⚙️ Mekanik / Sağ El Kuralı Sorusu

**Soru Karakteristiği:**
Vida ve somun sistemlerinde sağ el kuralı kullanımı.

| Sonuç | Değer |
|---|---|
| Model Tahmini | B |
| Gerçek Cevap | B |
| Durum | ✅ DOĞRU |

### 🌌 Kozmoloji / Evrenin Genişlemesi Sorusu

**Soru Karakteristiği:**
Kırmızıya kayma ve evrenin genişleme ilişkisi.

| Sonuç | Değer |
|---|---|
| Model Tahmini | E |
| Gerçek Cevap | E |
| Durum | ✅ DOĞRU |

---

# 🔍 4. TEKNİK HATA ANALİZİ (ERROR AUDIT)

## 4.1 A ve D Şıkkı Dominasyonu

Model:

- A sınıfında Recall = 1.00
- D sınıfında Recall = 1.00

değerine ulaşmıştır.

Bu durum modelin belirli prompt yapılarında yüksek güvenli karar verme eğiliminde olduğunu göstermektedir.

---

## 4.2 E Şıkkı Recall Problemi

| Metrik | Sonuç |
|---|---|
| Precision | 1.00 |
| Recall | 0.33 |

Model yalnızca çok emin olduğu durumlarda E cevabını üretmiştir.

### Kök Sebep Analizi

Aşağıdaki durumlar E sınıfı performansını düşürmüştür:

- uzun akademik terminoloji,
- çok katmanlı bilimsel muhakeme,
- yüksek token yoğunluğu,
- positional bias.

---

## 4.3 Önerilen Çözümler

### Position Augmentation

Şıkların sırasını karıştırarak veri çoğaltma:

```text
A ↔ D
B ↔ E
```

### Synthetic QA Generation

LLM kullanılarak yeni bilimsel soru üretimi.

### Retrieval-Augmented Generation (RAG)

Wikipedia + arXiv tabanlı retrieval sistemi entegrasyonu.

---

# ⚙️ 5. KULLANILAN TEKNOLOJİLER

| Teknoloji | Kullanım |
|---|---|
| Python 3.12 | Ana programlama dili |
| PyTorch | Derin öğrenme altyapısı |
| Transformers | LLM framework |
| PEFT | LoRA adaptör eğitimi |
| BitsAndBytes | 4-bit quantization |
| Hugging Face | Model yönetimi |
| Kaggle | GPU eğitim ortamı |

---


## 🚀 6. CANLI UYGULAMA VE DENEYİM (LIVE DEMO)

Bu proje, son kullanıcıların ve araştırmacıların modelleri anlık olarak test edebilmesi için **Streamlit** altyapısı kullanılarak bir web uygulamasına dönüştürülmüştür. 

Eğitilmiş olan **Qwen2.5-1.5B QLoRA** akademisyen modelimizi harici bir kuruluma ihtiyaç duymadan, doğrudan tarayıcınız üzerinden test etmek için aşağıdaki bağlantıyı kullanabilirsiniz:

🔗 **[Hugging Face Spaces - Canlı Deneme Platformu](https://huggingface.co/spaces/sancos/llm-science-model)**

---



# 📁 7. PROJE KLASÖR YAPISI

```text
llm_science_exam/
│
├── llm_science_model/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── outputs/
│
├── notebooks/
│
├── submission.csv
│
├── README.md
│
└── requirements.txt
```

---

# 🚀 8. KURULUM VE ÇALIŞTIRMA

## 8.1 Kütüphanelerin Kurulumu

```bash
pip install -U transformers
pip install -U peft
pip install -U accelerate
pip install -U bitsandbytes
pip install -U datasets
pip install -U trl
pip install -U sentencepiece
```

---

## 8.2 Eğitim Başlatma

```bash
python train.py
```

---

## 8.3 Inference

```bash
python inference.py
```

---

# 🧠 9. GELECEK FAZ PLANLAMASI

İlerleyen sürümlerde aşağıdaki geliştirmeler planlanmaktadır:

- Retrieval-Augmented Generation (RAG)
- Dense Retrieval
- CrossEncoder Re-ranking
- MAP@3 Optimization
- Ensemble Learning
- Synthetic QA Expansion
- Self-Consistency Voting
- Multi-Agent Reasoning
- Flash Attention
- vLLM Inference

---

# 🖋️ 10. SONUÇ VE DEĞERLENDİRME

Bu proje kapsamında:

- düşük VRAM ortamında çalışan,
- modern üretken yapay zeka tabanlı,
- QLoRA optimize edilmiş,
- akademik reasoning yeteneği geliştirilmiş

bir bilimsel soru çözüm sistemi başarıyla geliştirilmiştir.

Sistem:

- Kaggle T4 GPU sınırları içerisinde stabil çalışmış,
- quantization teknikleriyle optimize edilmiş,
- klasik encoder tabanlı sistemlere kıyasla ciddi performans artışı sağlamıştır.

---

# 👨‍💻 11. HAZIRLAYAN

## Serdar Önal

Uzman İnşaat Mühendisi  &  Yapay Zeka Uygulayıcısı


