import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Buat folder sementara jika belum ada
if not os.path.exists("tempDir"):
    os.makedirs("tempDir")

# Load model
model = load_model('model.h5')

# Daftar kelas sesuai urutan pelatihan (EDIT jika urutan beda)
soil_types = ['humus', 'andosol', 'inceptisol', 'entisol', 'aluvial', 'pasir', 'laterit', 'latosol']

# Fungsi untuk memprediksi jenis tanah
def predict_soil_type(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction), np.max(prediction) * 100

# Rekomendasi tanaman
def get_recommendations(soil_type):
    recommendations = {
        'humus': ['Padi', 'Jagung'],
        'andosol': ['Kopi', 'Cokelat'],
        'inceptisol': ['Sayuran'],
        'entisol': ['Umbi-umbian'],
        'aluvial': ['Tebu'],
        'pasir': ['Kedelai'],
        'laterit': ['Teh'],
        'latosol': ['Buah-buahan']
    }
    return recommendations.get(soil_type.lower(), [])

# Streamlit UI
st.title("📸 Klasifikasi Tanah dan Rekomendasi Tanaman")
st.write("Unggah gambar tanah untuk mendapatkan klasifikasi dan rekomendasi tanaman.")

uploaded_file = st.file_uploader("📁 Pilih gambar tanah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = os.path.join("tempDir", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(img_path, caption='📷 Gambar yang diunggah', use_column_width=True)

    index, confidence = predict_soil_type(img_path)
    soil_type = soil_types[index]
    
    st.success(f"✅ Jenis Tanah: {soil_type.capitalize()} ({confidence:.2f}%)")
    recommendations = get_recommendations(soil_type)
    st.info(f"🌱 Rekomendasi tanaman: {', '.join(recommendations)}")
