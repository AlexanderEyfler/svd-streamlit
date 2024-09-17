import io as std_io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from skimage import io
from PIL import Image


# задаем параметры отображения страницы, иконку и тектовое наполнение
st.set_page_config(page_title='SVD basic', page_icon='👻')
st.title(
    'SVD разложение ЧБ изображения'
    )
st.markdown("""
### Basic task
Загрузи черно-белое (и только!) изображение, введи желаемое количество
сингулярных чисел, получи результат, по желанию - сохрани.
""")

upload_file = st.file_uploader(
    label='Загрузите ЧБ изображение в формате PNG, JPG или JPEG',
    type=['png', 'jpg', 'jpeg']
    )
if upload_file is not None:
    # берем только красный канал
    image = io.imread(upload_file)[:, :, 0]

    # разложим на матрицу SVD
    U, singular_values, V = np.linalg.svd(image)

    # создаем сразу диагональную матрицу
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(a=sigma, val=singular_values)

    # выбираем топ k сингулярных чисел
    top_k = st.number_input(
        'Введите (целое число) сингулярных чисел, по умолчанию считается половина от их общего числа:',
        min_value=1,
        max_value=min(image.shape),
        value=round(min(image.shape) / 2),
        step=10,
        format='%i'
        )

    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    
    result_image = trunc_U @ trunc_sigma  @trunc_V

    st.write('#### Визуализация работы сингулярного разложения (SVD)')
    fig, axes = plt.subplots(1, 2, figsize=(15,8))
    axes[0].imshow(image, cmap='gray')
    axes[1].imshow(result_image, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[1].set_title(f'Изображение по топ {top_k} сингулярным числам')
    plt.tight_layout();
    st.pyplot(fig=fig)
    
    # сохранение полученного изображения (графика сравнения картинок)
    fig.savefig('result_original_and_svg.png')

    with open('result_original_and_svg.png', 'rb') as file:
        button = st.download_button(
            label='Сохранить сравнительный график',
            data=file,
            file_name='result_original_and_svg.png',
            mime='image/png'
        )
    
    # сохранение только полученного изображения (без графика)
    # нормализуем изображение для корректного сохранения
    result_image_normalized = (result_image - result_image.min()) / (
        result_image.max() - result_image.min()
        )
    result_image_uint8 = (result_image_normalized * 255).astype(np.uint8)

    # Конвертируем в изображение PIL
    img = Image.fromarray(result_image_uint8)
    
    # Сохраняем изображение в буфер
    buf = std_io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    st.download_button(
    label='Сохранить прогнанное через SVD разложение изображение',
    data=buf,
    file_name='result_svd.png',
    mime='image/png'
)

else:
    st.stop()
