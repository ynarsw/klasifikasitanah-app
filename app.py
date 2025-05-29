import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model
model = load_model('model.h5')

# Fungsi untuk memprediksi jenis tanah
def predict_soil_type(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Rekomendasi tanaman berdasarkan jenis tanah
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
st.title("Klasifikasi Tanah dan Rekomendasi Tanaman")
st.write("Unggah gambar tanah untuk mendapatkan klasifikasi dan rekomendasi tanaman.")

uploaded_file = st.file_uploader("Pilih gambar tanah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan gambar sementara
    img_path = os.path.join("tempDir", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Tampilkan gambar
    st.image(img_path, caption='Gambar yang diunggah', use_column_width=True)
    
    # Prediksi
    soil_type_index = predict_soil_type(img_path)
    soil_types = list(model.class_indices.keys())
    soil_type = soil_types[soil_type_index]
    
    # Tampilkan hasil
    st.write(f"✅ Jenis Tanah: {soil_type.capitalize()}")
    
    # Rekomendasi tanaman
    recommendations = get_recommendations(soil_type)
    st.write(f"🌱 Rekomendasi tanaman: {', '.join(recommendations)}")
