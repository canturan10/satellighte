import streamlit as st
from PIL import Image
import numpy as np
import satellighte as sat


def main():
    # pylint: disable=no-member

    st.title("Satellighte Demo")
    st.write("Satellite Image Classification")

    image_file = st.file_uploader(
        "Upload image",
        type=["jpeg", "png", "jpg", "webp"],
    )

    if image_file:
        image = Image.open(image_file)
        if st.button("Process"):

            option = st.selectbox("Select model", sat.available_models())
            st.write("Selected Model:", option)

            image = np.array(image.convert("RGB"))
            FRAME_WINDOW = st.image([])
            model = sat.Classifier.from_pretrained(option)
            model.eval()

            results = model.predict(image)
            pil_img = sat.utils.visualize(image, results)
            pil_img.show()
            st.write(results)
            FRAME_WINDOW.image(pil_img)


if __name__ == "__main__":
    main()
