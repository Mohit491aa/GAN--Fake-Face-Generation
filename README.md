# 🧠 GAN-Powered Face Image Generation (128×128 Resolution)

This project demonstrates a **Generative Adversarial Network (GAN)** built using **TensorFlow** and **Keras**, capable of generating realistic human face images at a resolution of 128×128 pixels. The model learns to convert random noise vectors into detailed, symmetric facial images. After training for **105 epochs**, it produces visually compelling outputs with natural textures and structure.

<img width="721" height="662" alt="Screenshot 2025-07-27 at 11 32 15 PM" src="https://github.com/user-attachments/assets/be7a1e4c-9507-470e-9747-cf841e8e41ad" />

---
```
#📂 GAN-Face-Generator/
├── main.py # Generates sample outputs using the trained model
├── main.ipynb # Model building and training logic (Generator + Discriminator)
├── requirements.txt # List of required Python packages
├── .gitignore # Ignores virtual environments, checkpoints, caches, etc.
└── README.md # Project overview and documentation
```
----
## 📦 Installation & Dependencies

To set up the environment, install the required packages: `requirements.txt`:

```bash
pip install -r requirements.txt
```
Core Libraries Used:
	•	tensorflow
	•	numpy
	•	matplotlib
	•	tqdm
	•	time, os (standard libraries)

# 🧠 Model Architecture

Generator (Upsampling: 100-dim → 128×128×3)
- Dense(8×8×512) → Reshape → BatchNorm → LeakyReLU
- Conv2DTranspose: 256 → 128 → 64 → 32 → 3
- Final activation: tanh (outputs in [-1, 1])

Discriminator (Downsampling: 128×128×3 → binary output)
- Conv2D: 64 → 128 → 256 → 512
- LeakyReLU + Dropout after each layer
- Flatten → Dense(1) logit output

## 🧪 Training Strategy

The models were trained adversarially using a custom training loop in TensorFlow:
```python
#Loss Functions:
#Binary crossentropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output), real_output) + \
           cross_entropy(tf.zeros_like(fake_output), fake_output)

# Training Step:
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    ...
    g_loss = generator_loss(fake_output)
    d_loss = discriminator_loss(real_output, fake_output)
    ...
    # Apply gradients to both networks
    
# Optimizers:

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)
```
```txt
# 📈 Training Configuration
PParameter            | Value
---------------------|----------------------
Dataset              | CelebA
Noise Input          | 100-dimensional
Image Output         | 128×128 RGB
Epochs               | 105
Batch Size           | 128 (customizable)
Loss Function        | Binary Crossentropy
Optimizer            | Adam (learning rate = 1e-4)
Framework            | TensorFlow 2.16.2
```
# 🙌 Acknowledgements
	•	Ian Goodfellow et al., GAN Paper (2014)
	•	TensorFlow & Keras Teams
	•	CelebA Dataset


