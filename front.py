import requests
import streamlit as st

API_URL = 'http://127.0.0.1:8000/sports_predict/'

st.title('Классификация спорта')
st.caption('PyTorch, 100 видов спорта')

image_file = st.file_uploader('Загрузите фото', type=['jpg', 'jpeg', 'png', 'webp'])

if image_file:
    st.image(image_file, width='stretch')

    if st.button('Определить'):
        try:
            response = requests.post(
                API_URL,
                files={'file': (image_file.name, image_file.getvalue(), image_file.type)},
            )
        except requests.ConnectionError:
            st.error('FastAPI не запущен: python main.py')
        else:
            if response.status_code == 200:
                result = response.json()
                st.success(f"{result['label']} - {result['confidence']}%")
                st.table(result['top3'])
            else:
                st.error(response.json()['detail'])
else:
    st.info('Загрузите фото и нажмите «Определить»')
