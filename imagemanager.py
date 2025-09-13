import streamlit as st
from PIL import Image
import io
import zipfile

st.title("Image Renamer & ZIP Downloader")

# Step 1: User provides a base name
base_name = st.text_input("Enter the base name for your files:", value="image")

# Step 2: User uploads multiple images
uploaded_files = st.file_uploader(
    "Upload images", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Prepare in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            # Validate file
            if not validate_file(uploaded_file):
                continue
            # Generate new file name
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            new_name = f"{base_name}_{idx}{extension}"
            # Read image and save to bytes
            image = Image.open(uploaded_file)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG' if extension == '.png' else 'JPEG')
            img_bytes.seek(0)
            # Write to ZIP
            zip_file.writestr(new_name, img_bytes.read())
    zip_buffer.seek(0)

    st.success("All files have been renamed and zipped!")
    st.download_button(
        label="Download ZIP",
        data=zip_buffer,
        file_name=f"{base_name}_images.zip",
        mime="application/zip"
    )
