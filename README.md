# CNN Flask Image Classifier

Aplikasi web sederhana untuk klasifikasi gambar menggunakan **Convolutional Neural Network (CNN)** dan **Flask**.
User dapat mengunggah gambar, lalu sistem akan memprediksi kelas gambar tersebut.

---

## Features

* Upload gambar melalui web
* Prediksi menggunakan model CNN
* Menampilkan hasil klasifikasi + confidence
* Tampilan web sederhana (Bootstrap)

---

## Requirements

* Python 3.10
* pip

---

## Installation

1. Clone repository:

```bash
git clone https://github.com/USERNAME/cnn-flask-app.git
cd cnn-flask-app
```

2. Buat virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train Model

Sebelum menjalankan aplikasi, lakukan training model terlebih dahulu:

```bash
python train_model.py
```

Model akan disimpan di:

```
model/model.h5
```

---

## Run Application

Jalankan aplikasi Flask:

```bash
python app.py
```

Buka browser:

```
http://127.0.0.1:5000
```

---

## Usage

1. Upload gambar dari komputer
2. Klik tombol **Prediksi**
3. Lihat hasil klasifikasi dan confidence

---

## Notes

* File model (`model.h5`) tidak disertakan dalam repository
* Pastikan melakukan training terlebih dahulu sebelum menjalankan aplikasi
* Folder `static/uploads` digunakan untuk menyimpan gambar sementara

---

## Model Architecture

Model CNN terdiri dari beberapa layer:

* Convolutional Layer (Conv2D)
* MaxPooling Layer
* Fully Connected Layer (Dense)
* Dropout (untuk mengurangi overfitting)

---

## Technologies

* Python
* TensorFlow / Keras
* Flask
* Bootstrap

---

## Project Structure

```
cnn-flask-app/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   └── model.h5
│
├── static/
│   └── uploads/
│
└── templates/
    └── index.html
```

---

## Author

Nama: (Group 5)
Project: Team Assignment 2

---
