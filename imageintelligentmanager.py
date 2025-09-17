import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import io
import zipfile
import sqlite3
import uuid
import os
import base64
import mimetypes
import torch
import numpy as np
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError as e:
    st.warning(f"Background removal is disabled: {str(e)}. Install 'rembg' for this feature.")
    REMBG_AVAILABLE = False

# Try to import diffusers for inpainting
try:
    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Inpainting features (mole/cloth removal) are disabled: {str(e)}. Install 'diffusers torch transformers accelerate' for these features.")
    DIFFUSERS_AVAILABLE = False

DB_PATH = "image_manager.db"
MAX_FILE_SIZE_MB = 5
ALLOWED_TYPES = ['image/png', 'image/jpeg', '.png', '.jpg', '.jpeg']

# -------------------------------
# Helper Functions
# -------------------------------
def image_to_base64(image_data):
    """Convert image data (bytes) to base64 string."""
    return base64.b64encode(image_data).decode('utf-8') if isinstance(image_data, bytes) else image_data.encode('utf-8')

def validate_file(file):
    """Validate uploaded file size and type."""
    file_size_bytes = len(file.getvalue())
    if file_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File '{getattr(file, 'name', 'unknown')}' exceeds {MAX_FILE_SIZE_MB}MB limit.")
        return False
    file_type = file.type if hasattr(file, 'type') and file.type else os.path.splitext(getattr(file, 'name', ''))[1].lower()
    if file_type not in ALLOWED_TYPES:
        st.error(f"File '{getattr(file, 'name', 'unknown')}' must be PNG or JPG.")
        return False
    try:
        file.seek(0)
        Image.open(file).verify()
        file.seek(0)
    except Exception as e:
        st.error(f"File '{getattr(file, 'name', 'unknown')}' is invalid or corrupted: {str(e)}")
        return False
    return True

def extract_zip(zip_file, folder):
    """Extract images from a ZIP file to the database."""
    uploaded_count = 0
    try:
        zip_buffer = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_data = zip_ref.read(file_name)
                    file_buffer = io.BytesIO(file_data)
                    file_buffer.name = file_name
                    if validate_file(file_buffer):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        extension = os.path.splitext(file_name)[1].lower()
                        new_name = f"{uuid.uuid4()}{extension}"
                        c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, new_name))
                        if c.fetchone()[0] == 0:
                            c.execute("INSERT INTO images (name, folder, image_data) VALUES (?, ?, ?)",
                                      (new_name, folder, file_data))
                            uploaded_count += 1
                        conn.commit()
                        conn.close()
    except Exception as e:
        st.error(f"Error extracting ZIP: {str(e)}")
    return uploaded_count

def zoom_image(img, zoom_level, center_x, center_y):
    """Apply zoom to a PIL image and return the zoomed region."""
    width, height = img.size
    zoom_factor = max(1.0, zoom_level)
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    
    x0 = max(0, int(center_x * width - new_width // 2))
    y0 = max(0, int(center_y * height - new_height // 2))
    x1 = min(width, x0 + new_width)
    y1 = min(height, y0 + new_height)
    
    if x1 - x0 < new_width:
        x0 = max(0, x1 - new_width)
    if y1 - y0 < new_height:
        y0 = max(0, y1 - new_height)
    
    cropped = img.crop((x0, y0, x1, y1))
    zoomed = cropped.resize((width, height), Image.LANCZOS)
    return zoomed, (x0, y0, x1-x0, y1-y0)

def crop_image(img, crop_box):
    """Crop a PIL image using the provided box (x, y, w, h)."""
    x, y, w, h = crop_box
    return img.crop((x, y, x+w, y+h))

def rotate_image(img, angle):
    """Rotate a PIL image by the specified angle (degrees)."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def adjust_brightness(img, factor):
    """Adjust brightness of a PIL image (factor: 0.0 to 2.0)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img, factor):
    """Adjust contrast of a PIL image (factor: 0.0 to 2.0)."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def remove_background(img):
    """Remove background from a PIL image using rembg."""
    if not REMBG_AVAILABLE:
        return img
    try:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        output = remove(img_bytes.read())
        return Image.open(io.BytesIO(output)).convert('RGBA')
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}")
        return img

def apply_transformations(img, zoom_level, center_x, center_y, crop_box, rotation, brightness, contrast):
    """Apply all transformations (zoom, crop, rotation, brightness, contrast) for preview."""
    processed = img.copy()
    crop_coords = None
    if crop_box:
        processed = crop_image(processed, crop_box)
    else:
        processed, crop_coords = zoom_image(processed, zoom_level, center_x, center_y)
    processed = rotate_image(processed, rotation)
    processed = adjust_brightness(processed, brightness)
    processed = adjust_contrast(processed, contrast)
    return processed, crop_coords

def draw_crop_overlay(img, crop_box):
    """Draw crop border on a copy of the image."""
    overlay_img = img.copy()
    if crop_box:
        draw = ImageDraw.Draw(overlay_img)
        x, y, w, h = crop_box
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=5)
    return overlay_img

def inpaint_image(img, mask_img, prompt="a seamless image", num_inference_steps=20, guidance_scale=7.5):
    """Perform inpainting using Stable Diffusion Inpainting model."""
    if not DIFFUSERS_AVAILABLE:
        st.error("Diffusers not available. Please install required packages.")
        return img

    try:
        # Load the pipeline if not already loaded
        if "inpaint_pipeline" not in st.session_state:
            st.session_state.inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
            )
            st.session_state.inpaint_pipeline.enable_model_cpu_offload()
            if hasattr(st.session_state.inpaint_pipeline, 'enable_xformers_memory_efficient_attention'):
                st.session_state.inpaint_pipeline.enable_xformers_memory_efficient_attention()

        pipeline = st.session_state.inpaint_pipeline

        # Convert PIL to the format expected (resize if necessary)
        init_image = img.resize((512, 512))
        mask_image = mask_img.resize((512, 512)).convert('L')  # Grayscale mask

        # Perform inpainting
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(np.random.randint(0, 1000))
        output = pipeline(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Resize back to original size
        output = output.resize(img.size)
        return output

    except Exception as e:
        st.error(f"Inpainting failed: {str(e)}")
        return img

def save_image_to_db(folder, name, img):
    """Save a processed image to the database."""
    img_bytes = io.BytesIO()
    img_format = 'PNG' if img.mode == 'RGBA' else 'JPEG'
    img.save(img_bytes, format=img_format)
    img_bytes.seek(0)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE images SET image_data = ? WHERE folder = ? AND name = ?",
              (img_bytes.read(), folder, name))
    conn.commit()
    conn.close()

def init_db():
    """Initialize SQLite database with folders and images tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            folder TEXT NOT NULL,
            image_data BLOB NOT NULL,
            label TEXT,
            base_name TEXT,
            sequence INTEGER,
            FOREIGN KEY(folder) REFERENCES folders(folder)
        )
    """)
    default_folders = [
        {"name": "General", "description": "General images", "folder": "general"},
        {"name": "Classified", "description": "Classified images", "folder": "classified"},
    ]
    for folder_data in default_folders:
        c.execute("SELECT COUNT(*) FROM folders WHERE folder = ?", (folder_data["folder"],))
        if c.fetchone()[0] == 0:
            c.execute("""
                INSERT INTO folders (folder, name, description)
                VALUES (?, ?, ?)
            """, (folder_data["folder"], folder_data["name"], folder_data["description"]))
    conn.commit()
    conn.close()

def load_folders():
    """Load all folders from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT folder, name, description FROM folders")
    folders = [{"folder": r[0], "name": r[1], "description": r[2]} for r in c.fetchall()]
    conn.close()
    return folders

def add_folder(folder, name, description):
    """Add a new folder to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO folders (folder, name, description)
            VALUES (?, ?, ?)
        """, (folder, name, description or ""))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error(f"Folder '{folder}' already exists.")
        return False
    except Exception as e:
        st.error(f"Error adding folder: {str(e)}")
        return False

def load_images_to_db(uploaded_files, folder, base_name):
    """Load uploaded images to the database with sequential naming."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uploaded_count = 0
    c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (folder, base_name))
    max_sequence = c.fetchone()[0] or 0
    for idx, uploaded_file in enumerate(uploaded_files, start=max_sequence + 1):
        if validate_file(uploaded_file):
            image_data = uploaded_file.read()
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            new_name = f"{base_name}_{idx}{extension}"
            c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, new_name))
            if c.fetchone()[0] == 0:
                c.execute("INSERT INTO images (name, folder, image_data, base_name, sequence) VALUES (?, ?, ?, ?, ?)",
                          (new_name, folder, image_data, base_name, idx))
                uploaded_count += 1
    conn.commit()
    conn.close()
    return uploaded_count

def get_images(folder):
    """Retrieve images and metadata for a folder."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, image_data, label, base_name, sequence FROM images WHERE folder = ?", (folder,))
    images = []
    for r in c.fetchall():
        name, data, label, base_name, sequence = r
        try:
            img = Image.open(io.BytesIO(data))
            base64_image = image_to_base64(data)
            images.append({
                "name": name,
                "image": img,
                "data": data,
                "label": label or "",
                "base_name": base_name or "",
                "sequence": sequence,
                "base64": base64_image
            })
        except Exception as e:
            st.error(f"Error loading image {name}: {str(e)}")
    conn.close()
    return images

def update_image_label(folder, name, label):
    """Update the label for an image."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE images SET label = ? WHERE folder = ? AND name = ?", (label, folder, name))
    conn.commit()
    conn.close()

def classify_image(folder, name, new_folder, new_base_name):
    """Move an image to a new folder with a new base name."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_data, label FROM images WHERE folder = ? AND name = ?", (folder, name))
    result = c.fetchone()
    if result:
        image_data, label = result
        c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (new_folder, new_base_name))
        max_sequence = c.fetchone()[0] or 0
        new_sequence = max_sequence + 1
        extension = os.path.splitext(name)[1].lower()
        new_name = f"{new_base_name}_{new_sequence}{extension}"
        c.execute("INSERT INTO images (name, folder, image_data, label, base_name, sequence) VALUES (?, ?, ?, ?, ?, ?)",
                  (new_name, new_folder, image_data, label, new_base_name, new_sequence))
        c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
        conn.commit()
    conn.close()

def delete_image(folder, name):
    """Delete an image from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
    conn.commit()
    conn.close()

def create_zip(folder, base_name):
    """Create a ZIP file for all images in a folder with the given base name."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        images = get_images(folder)
        for img_dict in images:
            if img_dict["base_name"] == base_name:
                zip_file.writestr(img_dict["name"], img_dict["data"])
    zip_buffer.seek(0)
    return zip_buffer

# -------------------------------
# Initialize DB & Session State
# -------------------------------
init_db()
if "zoom_folder" not in st.session_state:
    st.session_state.zoom_folder = None
if "zoom_index" not in st.session_state:
    st.session_state.zoom_index = 0
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 1.0
if "zoom_center_x" not in st.session_state:
    st.session_state.zoom_center_x = 0.5
if "zoom_center_y" not in st.session_state:
    st.session_state.zoom_center_y = 0.5
if "crop_box" not in st.session_state:
    st.session_state.crop_box = None
if "rotation" not in st.session_state:
    st.session_state.rotation = 0.0
if "brightness" not in st.session_state:
    st.session_state.brightness = 1.0
if "contrast" not in st.session_state:
    st.session_state.contrast = 1.0
if "preview_image" not in st.session_state:
    st.session_state.preview_image = None
if "crop_preview_image" not in st.session_state:
    st.session_state.crop_preview_image = None
if "inpaint_pipeline" not in st.session_state:
    st.session_state.inpaint_pipeline = None

# -------------------------------
# Sidebar: Admin Controls & Upload
# -------------------------------
with st.sidebar:
    st.title("Admin Login")
    with st.form(key="login_form"):
        pwd = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Login", key="login_button"):
            if pwd == "admin123":
                st.session_state.is_admin = True
                st.success("Logged in as admin!")
            else:
                st.error("Incorrect password")
    if st.session_state.is_admin and st.button("Logout", key="logout_button"):
        st.session_state.is_admin = False
        st.success("Logged out")
        st.rerun()

    if st.session_state.is_admin:
        st.subheader("Manage Folders & Images")
        with st.form(key="add_folder_form"):
            new_folder = st.text_input("Folder Name (e.g., 'newfolder')", key="new_folder_input")
            new_name = st.text_input("Display Name", key="new_name_input")
            new_description = st.text_area("Description (optional)", key="new_description_input")
            if st.form_submit_button("Add Folder", key="add_folder_button"):
                if new_folder and new_name:
                    if add_folder(new_folder.lower(), new_name, new_description):
                        st.success(f"Folder '{new_folder}' added successfully!")
                        st.rerun()
                else:
                    st.error("Folder Name and Display Name are required.")

        st.subheader("Upload Images or ZIP")
        data = load_folders()
        folder_choice = st.selectbox("Select Folder", [item["folder"] for item in data], key="upload_folder_select")
        base_name = st.text_input("Enter base name for files:", value="image", key="base_name_input")
        uploaded_files = st.file_uploader(
            "Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload_files"
        )
        uploaded_zip = st.file_uploader("Upload ZIP", type=['zip'], key="upload_zip")
        if st.button("Upload Images", key="upload_images_button") and uploaded_files:
            uploaded_count = load_images_to_db(uploaded_files, folder_choice, base_name)
            st.success(f"{uploaded_count} image(s) uploaded to '{folder_choice}'!")
        if st.button("Upload ZIP", key="upload_zip_button") and uploaded_zip:
            uploaded_count = extract_zip(uploaded_zip, folder_choice)
            st.success(f"{uploaded_count} image(s) extracted to '{folder_choice}'!")

# -------------------------------
# CSS Styling
# -------------------------------
st.markdown("""
<style>
.folder-card {background: #f9f9f9; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
.folder-header {font-size:1.5em; color:#333; margin-bottom:10px;}
.image-grid {display:flex; flex-wrap:wrap; gap:10px;}
img {border-radius:4px; max-width:100px; object-fit:cover;}
.image-info {margin-top:10px; word-break:break-all; font-size:0.9em;}
.warning {color: red; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Main App UI
# -------------------------------
st.title("📸 Image Manager & Editor")

data = load_folders()
if st.session_state.zoom_folder is None:
    # Grid View
    if not data:
        st.info("No folders available. Admins can create folders in the sidebar.")
    for f in data:
        st.markdown(
            f'<div class="folder-card"><div class="folder-header">'
            f'{f["name"]} ({f["description"] or "No description"})</div>',
            unsafe_allow_html=True
        )
        images = get_images(f["folder"])
        if images:
            # ZIP Download
            base_names = list(set(img["base_name"] for img in images if img["base_name"]))
            if base_names:
                zip_base_name = st.selectbox(
                    f"Select base name for ZIP ({f['name']})",
                    base_names,
                    key=f"zip_base_name_{f['folder']}"
                )
                if st.button(f"Download ZIP ({f['name']})", key=f"zip_download_{f['folder']}"):
                    zip_buffer = create_zip(f["folder"], zip_base_name)
                    st.download_button(
                        label=f"Download {zip_base_name}_images.zip",
                        data=zip_buffer,
                        file_name=f"{zip_base_name}_images.zip",
                        mime="application/zip",
                        key=f"zip_download_button_{f['folder']}"
                    )
            # Image Grid
            cols = st.columns(4)
            for idx, img_dict in enumerate(images):
                with cols[idx % 4]:
                    if st.button("🔍 Edit", key=f"edit_{f['folder']}_{idx}"):
                        st.session_state.zoom_folder = f["folder"]
                        st.session_state.zoom_index = idx
                        st.session_state.zoom_level = 1.0
                        st.session_state.zoom_center_x = 0.5
                        st.session_state.zoom_center_y = 0.5
                        st.session_state.crop_box = None
                        st.session_state.rotation = 0.0
                        st.session_state.brightness = 1.0
                        st.session_state.contrast = 1.0
                        st.session_state.preview_image = None
                        st.session_state.crop_preview_image = None
                        st.rerun()
                    st.image(img_dict["image"], use_container_width=True, caption=f"Photo {idx+1}")
                    st.markdown(f'<div class="image-info"><b>Name:</b> {img_dict["name"]}<br><b>Label:</b> {img_dict["label"]}</div>',
                                unsafe_allow_html=True)
                    if st.session_state.is_admin:
                        if st.button("🗑️ Delete", key=f"delete_grid_{f['folder']}_{img_dict['name']}"):
                            delete_image(f["folder"], img_dict["name"])
                            st.success("Image deleted.")
                            st.rerun()
        else:
            st.warning(f"No images in '{f['name']}'")

else:
    # Zoom/Edit View
    folder = st.session_state.zoom_folder
    images = get_images(folder)
    idx = st.session_state.zoom_index
    if idx >= len(images):
        idx = 0
        st.session_state.zoom_index = 0
    img_dict = images[idx]

    st.subheader(f"🔍 Editing {folder} ({idx+1}/{len(images)})")

    # Image Editing Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        # Display Image (Preview or Original)
        if st.session_state.preview_image:
            st.image(st.session_state.preview_image, use_container_width=True, caption="Preview")
        else:
            st.image(img_dict["image"], use_container_width=True, caption="Original Image")
        # Display Crop Preview if available
        if st.session_state.crop_preview_image:
            st.image(st.session_state.crop_preview_image, use_container_width=True, caption="Crop Preview with Borders")
    with col2:
        # Zoom Controls
        st.subheader("Zoom")
        zoom_level = st.slider("Zoom Level", 1.0, 5.0, st.session_state.zoom_level, key=f"zoom_{folder}_{idx}")
        center_x = st.slider("Center X", 0.0, 1.0, st.session_state.zoom_center_x, key=f"center_x_{folder}_{idx}")
        center_y = st.slider("Center Y", 0.0, 1.0, st.session_state.zoom_center_y, key=f"center_y_{folder}_{idx}")

        # Crop Controls
        st.subheader("Crop")
        st.markdown("Crop X and Crop Y are the coordinates starting at the top-left corner (0,0). Crop Width and Height are the distances (in pixels) relative to these X and Y coordinates.")
        crop_x = st.slider("Crop X", 0, img_dict["image"].width, 0, key=f"crop_x_{folder}_{idx}")
        crop_y = st.slider("Crop Y", 0, img_dict["image"].height, 0, key=f"crop_y_{folder}_{idx}")
        crop_w = st.slider("Crop Width", 1, img_dict["image"].width - crop_x, 100, key=f"crop_w_{folder}_{idx}")
        crop_h = st.slider("Crop Height", 1, img_dict["image"].height - crop_y, 100, key=f"crop_h_{folder}_{idx}")

        # Preview Crop Button
        if st.button("Preview Crop", key=f"preview_crop_{folder}_{idx}"):
            crop_box_temp = (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None
            st.session_state.crop_preview_image = draw_crop_overlay(img_dict["image"], crop_box_temp)
            st.rerun()

        # Rotation Control
        st.subheader("Rotation")
        rotation = st.slider("Rotation (degrees)", -180.0, 180.0, st.session_state.rotation, key=f"rotation_{folder}_{idx}")

        # Brightness and Contrast Controls
        st.subheader("Adjustments")
        brightness = st.slider("Brightness", 0.0, 2.0, st.session_state.brightness, key=f"brightness_{folder}_{idx}")
        contrast = st.slider("Contrast", 0.0, 2.0, st.session_state.contrast, key=f"contrast_{folder}_{idx}")

        # Preview Button
        if st.button("Preview", key=f"preview_{folder}_{idx}"):
            processed, _ = apply_transformations(
                img_dict["image"],
                zoom_level,
                center_x,
                center_y,
                (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None,
                rotation,
                brightness,
                contrast
            )
            st.session_state.preview_image = processed
            st.session_state.zoom_level = zoom_level
            st.session_state.zoom_center_x = center_x
            st.session_state.zoom_center_y = center_y
            st.session_state.crop_box = (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None
            st.session_state.rotation = rotation
            st.session_state.brightness = brightness
            st.session_state.contrast = contrast
            st.rerun()

        # Apply Changes Button
        if st.button("Apply Changes", key=f"apply_{folder}_{idx}"):
            processed, _ = apply_transformations(
                img_dict["image"],
                zoom_level,
                center_x,
                center_y,
                (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None,
                rotation,
                brightness,
                contrast
            )
            save_image_to_db(folder, img_dict["name"], processed)
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.success("Changes applied and saved to database.")
            st.rerun()

        # Reset Button
        if st.button("Reset", key=f"reset_{folder}_{idx}"):
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.rotation = 0.0
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.rerun()

        # Background Removal
        if st.button("Remove Background", key=f"remove_bg_{folder}_{idx}"):
            processed = remove_background(img_dict["image"])
            save_image_to_db(folder, img_dict["name"], processed)
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.success("Background removed and saved.")
            st.rerun()

        # Inpainting Controls
        st.subheader("AI Inpainting (Object Removal)")
        st.markdown("Upload a mask image (white areas will be inpainted). This uses Stable Diffusion Inpainting model. Requires GPU for best performance.")
        mask_upload = st.file_uploader("Upload Mask Image (PNG/JPG, white=remove)", type=['png', 'jpg', 'jpeg'], key=f"mask_{folder}_{idx}")
        if mask_upload:
            mask_img = Image.open(mask_upload)
            st.image(mask_img, caption="Mask", use_container_width=True)
        else:
            mask_img = None

        if DIFFUSERS_AVAILABLE and mask_img:
            prompt = st.text_input("Inpainting Prompt", value="a seamless image without the object", key=f"inpaint_prompt_{folder}_{idx}")
            if st.button("Apply Inpainting", key=f"inpaint_{folder}_{idx}"):
                processed = inpaint_image(img_dict["image"], mask_img, prompt)
                save_image_to_db(folder, img_dict["name"], processed)
                st.session_state.preview_image = None
                st.session_state.crop_preview_image = None
                st.success("Inpainting applied and saved.")
                st.rerun()

        # Mole Removal (using inpainting)
        st.markdown('<div class="warning">For mole removal: Create a mask highlighting the mole and use the inpainting feature above with prompt like "smooth skin without mole".</div>', unsafe_allow_html=True)

        # Cloth Removal (Strong Warning)
        st.markdown('<div class="warning">⚠️ Cloth removal raises serious ethical, privacy, and consent concerns. Use only on your own images with explicit consent. Misuse can lead to legal issues. Proceed at your own risk.</div>', unsafe_allow_html=True)
        if DIFFUSERS_AVAILABLE and mask_img:
            cloth_prompt = st.text_input("Cloth Removal Prompt (use responsibly)", value="nude body seamless", key=f"cloth_prompt_{folder}_{idx}")
            if st.button("Apply Cloth Removal (ETHICAL WARNING)", key=f"cloth_remove_{folder}_{idx}"):
                processed = inpaint_image(img_dict["image"], mask_img, cloth_prompt)
                save_image_to_db(folder, img_dict["name"], processed)
                st.session_state.preview_image = None
                st.session_state.crop_preview_image = None
                st.warning("Cloth removal applied. Remember ethical responsibilities.")
                st.rerun()
        else:
            st.warning("Upload a mask and install diffusers for cloth removal.")

    # Label and Classify
    label = st.text_input("Label", value=img_dict["label"], key=f"label_{folder}_{idx}")
    if st.button("Update Label", key=f"label_button_{folder}_{idx}"):
        update_image_label(folder, img_dict["name"], label)
        st.success("Label updated.")
        st.rerun()

    if st.session_state.is_admin:
        st.subheader("Classify Image")
        data = load_folders()
        new_folder = st.selectbox("Select Destination Folder", [item["folder"] for item in data], key=f"classify_folder_{folder}_{idx}")
        new_base_name = st.text_input("New Base Name", value="classified_image", key=f"classify_base_name_{folder}_{idx}")
        if st.button("Classify Image", key=f"classify_{folder}_{idx}"):
            classify_image(folder, img_dict["name"], new_folder, new_base_name)
            st.success(f"Image classified to '{new_folder}' as '{new_base_name}'.")
            st.session_state.zoom_index = max(0, idx-1)
            if len(get_images(folder)) == 0:
                st.session_state.zoom_folder = None
                st.session_state.zoom_index = 0
            st.rerun()

    # Download
    mime, _ = mimetypes.guess_type(img_dict["name"])
    st.download_button(
        "⬇️ Download",
        data=img_dict["data"],
        file_name=img_dict["name"],
        mime=mime,
        key=f"download_{folder}_{img_dict['name']}"
    )

    if st.session_state.is_admin:
        if st.button("🗑️ Delete Image", key=f"delete_zoom_{folder}_{img_dict['name']}"):
            delete_image(folder, img_dict["name"])
            st.success("Image deleted.")
            st.session_state.zoom_index = max(0, idx-1)
            if len(get_images(folder)) == 0:
                st.session_state.zoom_folder = None
                st.session_state.zoom_index = 0
            st.rerun()

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        if idx > 0 and st.button("◄ Previous", key=f"prev_{folder}_{idx}"):
            st.session_state.zoom_index -= 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.rotation = 0.0
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.rerun()
    with col3:
        if idx < len(images)-1 and st.button("Next ►", key=f"next_{folder}_{idx}"):
            st.session_state.zoom_index += 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.rotation = 0.0
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.rerun()

    if st.button("⬅️ Back to Grid", key=f"back_{folder}_{idx}"):
        st.session_state.zoom_folder = None
        st.session_state.zoom_index = 0
        st.session_state.zoom_level = 1.0
        st.session_state.zoom_center_x = 0.5
        st.session_state.zoom_center_y = 0.5
        st.session_state.crop_box = None
        st.session_state.rotation = 0.0
        st.session_state.brightness = 1.0
        st.session_state.contrast = 1.0
        st.session_state.preview_image = None
        st.session_state.crop_preview_image = None
        st.rerun()
``````python
import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw
import io
import zipfile
import sqlite3
import uuid
import os
import base64
import mimetypes
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError as e:
    st.warning(f"Background removal is disabled: {str(e)}. Install 'rembg' for this feature.")
    REMBG_AVAILABLE = False

try:
    from diffusers import AutoPipelineForInpainting
    import torch
    AI_AVAILABLE = True
except ImportError as e:
    st.warning(f"AI features disabled: {str(e)}. Install 'diffusers[torch]' for mole and cloth removal.")
    AI_AVAILABLE = False

DB_PATH = "image_manager.db"
MAX_FILE_SIZE_MB = 5
ALLOWED_TYPES = ['image/png', 'image/jpeg', '.png', '.jpg', '.jpeg']

# -------------------------------
# Helper Functions
# -------------------------------
def image_to_base64(image_data):
    """Convert image data (bytes) to base64 string."""
    return base64.b64encode(image_data).decode('utf-8') if isinstance(image_data, bytes) else image_data.encode('utf-8')

def validate_file(file):
    """Validate uploaded file size and type."""
    file_size_bytes = len(file.getvalue())
    if file_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File '{getattr(file, 'name', 'unknown')}' exceeds {MAX_FILE_SIZE_MB}MB limit.")
        return False
    file_type = file.type if hasattr(file, 'type') and file.type else os.path.splitext(getattr(file, 'name', ''))[1].lower()
    if file_type not in ALLOWED_TYPES:
        st.error(f"File '{getattr(file, 'name', 'unknown')}' must be PNG or JPG.")
        return False
    try:
        file.seek(0)
        Image.open(file).verify()
        file.seek(0)
    except Exception as e:
        st.error(f"File '{getattr(file, 'name', 'unknown')}' is invalid or corrupted: {str(e)}")
        return False
    return True

def extract_zip(zip_file, folder):
    """Extract images from a ZIP file to the database."""
    uploaded_count = 0
    try:
        zip_buffer = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_data = zip_ref.read(file_name)
                    file_buffer = io.BytesIO(file_data)
                    file_buffer.name = file_name
                    if validate_file(file_buffer):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        extension = os.path.splitext(file_name)[1].lower()
                        new_name = f"{uuid.uuid4()}{extension}"
                        c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, new_name))
                        if c.fetchone()[0] == 0:
                            c.execute("INSERT INTO images (name, folder, image_data) VALUES (?, ?, ?)",
                                      (new_name, folder, file_data))
                            uploaded_count += 1
                        conn.commit()
                        conn.close()
    except Exception as e:
        st.error(f"Error extracting ZIP: {str(e)}")
    return uploaded_count

def zoom_image(img, zoom_level, center_x, center_y):
    """Apply zoom to a PIL image and return the zoomed region."""
    width, height = img.size
    zoom_factor = max(1.0, zoom_level)
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    
    x0 = max(0, int(center_x * width - new_width // 2))
    y0 = max(0, int(center_y * height - new_height // 2))
    x1 = min(width, x0 + new_width)
    y1 = min(height, y0 + new_height)
    
    if x1 - x0 < new_width:
        x0 = max(0, x1 - new_width)
    if y1 - y0 < new_height:
        y0 = max(0, y1 - new_height)
    
    cropped = img.crop((x0, y0, x1, y1))
    zoomed = cropped.resize((width, height), Image.LANCZOS)
    return zoomed, (x0, y0, x1-x0, y1-y0)

def crop_image(img, crop_box):
    """Crop a PIL image using the provided box (x, y, w, h)."""
    x, y, w, h = crop_box
    return img.crop((x, y, x+w, y+h))

def rotate_image(img, angle):
    """Rotate a PIL image by the specified angle (degrees)."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def adjust_brightness(img, factor):
    """Adjust brightness of a PIL image (factor: 0.0 to 2.0)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img, factor):
    """Adjust contrast of a PIL image (factor: 0.0 to 2.0)."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def remove_background(img):
    """Remove background from a PIL image using rembg."""
    if not REMBG_AVAILABLE:
        return img
    try:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        output = remove(img_bytes.read())
        return Image.open(io.BytesIO(output)).convert('RGBA')
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}")
        return img

def inpaint_image(img, crop_box, prompt, negative_prompt="deformed, ugly, disfigured, poor details, bad anatomy"):
    """Inpaint the selected area using AI."""
    if not AI_AVAILABLE:
        st.warning("AI features are disabled. Please install 'diffusers[torch]'.")
        return img
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None
        )
        pipe.to(device)
        
        # Create mask
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        x, y, w, h = crop_box
        draw.rectangle((x, y, x + w, y + h), fill=255)
        
        # Inpaint
        result = pipe(
            prompt=prompt,
            image=img,
            mask_image=mask,
            strength=1.0,
            guidance_scale=7.5,
            negative_prompt=negative_prompt
        ).images[0]
        return result
    except Exception as e:
        st.warning(f"AI inpainting failed: {str(e)}")
        return img

def apply_transformations(img, zoom_level, center_x, center_y, crop_box, rotation, brightness, contrast):
    """Apply all transformations (zoom, crop, rotation, brightness, contrast) for preview."""
    processed = img.copy()
    crop_coords = None
    if crop_box:
        processed = crop_image(processed, crop_box)
    else:
        processed, crop_coords = zoom_image(processed, zoom_level, center_x, center_y)
    processed = rotate_image(processed, rotation)
    processed = adjust_brightness(processed, brightness)
    processed = adjust_contrast(processed, contrast)
    return processed, crop_coords

def draw_crop_overlay(img, crop_box):
    """Draw crop border on a copy of the image."""
    overlay_img = img.copy()
    if crop_box:
        draw = ImageDraw.Draw(overlay_img)
        x, y, w, h = crop_box
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=5)
    return overlay_img

def save_image_to_db(folder, name, img):
    """Save a processed image to the database."""
    img_bytes = io.BytesIO()
    img_format = 'PNG' if img.mode == 'RGBA' else 'JPEG'
    img.save(img_bytes, format=img_format)
    img_bytes.seek(0)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE images SET image_data = ? WHERE folder = ? AND name = ?",
              (img_bytes.read(), folder, name))
    conn.commit()
    conn.close()

def init_db():
    """Initialize SQLite database with folders and images tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            folder TEXT NOT NULL,
            image_data BLOB NOT NULL,
            label TEXT,
            base_name TEXT,
            sequence INTEGER,
            FOREIGN KEY(folder) REFERENCES folders(folder)
        )
    """)
    default_folders = [
        {"name": "General", "description": "General images", "folder": "general"},
        {"name": "Classified", "description": "Classified images", "folder": "classified"},
    ]
    for folder_data in default_folders:
        c.execute("SELECT COUNT(*) FROM folders WHERE folder = ?", (folder_data["folder"],))
        if c.fetchone()[0] == 0:
            c.execute("""
                INSERT INTO folders (folder, name, description)
                VALUES (?, ?, ?)
            """, (folder_data["folder"], folder_data["name"], folder_data["description"]))
    conn.commit()
    conn.close()

def load_folders():
    """Load all folders from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT folder, name, description FROM folders")
    folders = [{"folder": r[0], "name": r[1], "description": r[2]} for r in c.fetchall()]
    conn.close()
    return folders

def add_folder(folder, name, description):
    """Add a new folder to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO folders (folder, name, description)
            VALUES (?, ?, ?)
        """, (folder, name, description or ""))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error(f"Folder '{folder}' already exists.")
        return False
    except Exception as e:
        st.error(f"Error adding folder: {str(e)}")
        return False

def load_images_to_db(uploaded_files, folder, base_name):
    """Load uploaded images to the database with sequential naming."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uploaded_count = 0
    c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (folder, base_name))
    max_sequence = c.fetchone()[0] or 0
    for idx, uploaded_file in enumerate(uploaded_files, start=max_sequence + 1):
        if validate_file(uploaded_file):
            image_data = uploaded_file.read()
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            new_name = f"{base_name}_{idx}{extension}"
            c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, new_name))
            if c.fetchone()[0] == 0:
                c.execute("INSERT INTO images (name, folder, image_data, base_name, sequence) VALUES (?, ?, ?, ?, ?)",
                          (new_name, folder, image_data, base_name, idx))
                uploaded_count += 1
    conn.commit()
    conn.close()
    return uploaded_count

def get_images(folder):
    """Retrieve images and metadata for a folder."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, image_data, label, base_name, sequence FROM images WHERE folder = ?", (folder,))
    images = []
    for r in c.fetchall():
        name, data, label, base_name, sequence = r
        try:
            img = Image.open(io.BytesIO(data))
            base64_image = image_to_base64(data)
            images.append({
                "name": name,
                "image": img,
                "data": data,
                "label": label or "",
                "base_name": base_name or "",
                "sequence": sequence,
                "base64": base64_image
            })
        except Exception as e:
            st.error(f"Error loading image {name}: {str(e)}")
    conn.close()
    return images

def update_image_label(folder, name, label):
    """Update the label for an image."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE images SET label = ? WHERE folder = ? AND name = ?", (label, folder, name))
    conn.commit()
    conn.close()

def classify_image(folder, name, new_folder, new_base_name):
    """Move an image to a new folder with a new base name."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_data, label FROM images WHERE folder = ? AND name = ?", (folder, name))
    result = c.fetchone()
    if result:
        image_data, label = result
        c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (new_folder, new_base_name))
        max_sequence = c.fetchone()[0] or 0
        new_sequence = max_sequence + 1
        extension = os.path.splitext(name)[1].lower()
        new_name = f"{new_base_name}_{new_sequence}{extension}"
        c.execute("INSERT INTO images (name, folder, image_data, label, base_name, sequence) VALUES (?, ?, ?, ?, ?, ?)",
                  (new_name, new_folder, image_data, label, new_base_name, new_sequence))
        c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
        conn.commit()
    conn.close()

def delete_image(folder, name):
    """Delete an image from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
    conn.commit()
    conn.close()

def create_zip(folder, base_name):
    """Create a ZIP file for all images in a folder with the given base name."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        images = get_images(folder)
        for img_dict in images:
            if img_dict["base_name"] == base_name:
                zip_file.writestr(img_dict["name"], img_dict["data"])
    zip_buffer.seek(0)
    return zip_buffer

# -------------------------------
# Initialize DB & Session State
# -------------------------------
init_db()
if "zoom_folder" not in st.session_state:
    st.session_state.zoom_folder = None
if "zoom_index" not in st.session_state:
    st.session_state.zoom_index = 0
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 1.0
if "zoom_center_x" not in st.session_state:
    st.session_state.zoom_center_x = 0.5
if "zoom_center_y" not in st.session_state:
    st.session_state.zoom_center_y = 0.5
if "crop_box" not in st.session_state:
    st.session_state.crop_box = None
if "rotation" not in st.session_state:
    st.session_state.rotation = 0.0
if "brightness" not in st.session_state:
    st.session_state.brightness = 1.0
if "contrast" not in st.session_state:
    st.session_state.contrast = 1.0
if "preview_image" not in st.session_state:
    st.session_state.preview_image = None
if "crop_preview_image" not in st.session_state:
    st.session_state.crop_preview_image = None

# -------------------------------
# Sidebar: Admin Controls & Upload
# -------------------------------
with st.sidebar:
    st.title("Admin Login")
    with st.form(key="login_form"):
        pwd = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Login", key="login_button"):
            if pwd == "admin123":
                st.session_state.is_admin = True
                st.success("Logged in as admin!")
            else:
                st.error("Incorrect password")
    if st.session_state.is_admin and st.button("Logout", key="logout_button"):
        st.session_state.is_admin = False
        st.success("Logged out")
        st.rerun()

    if st.session_state.is_admin:
        st.subheader("Manage Folders & Images")
        with st.form(key="add_folder_form"):
            new_folder = st.text_input("Folder Name (e.g., 'newfolder')", key="new_folder_input")
            new_name = st.text_input("Display Name", key="new_name_input")
            new_description = st.text_area("Description (optional)", key="new_description_input")
            if st.form_submit_button("Add Folder", key="add_folder_button"):
                if new_folder and new_name:
                    if add_folder(new_folder.lower(), new_name, new_description):
                        st.success(f"Folder '{new_folder}' added successfully!")
                        st.rerun()
                else:
                    st.error("Folder Name and Display Name are required.")

        st.subheader("Upload Images or ZIP")
        data = load_folders()
        folder_choice = st.selectbox("Select Folder", [item["folder"] for item in data], key="upload_folder_select")
        base_name = st.text_input("Enter base name for files:", value="image", key="base_name_input")
        uploaded_files = st.file_uploader(
            "Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload_files"
        )
        uploaded_zip = st.file_uploader("Upload ZIP", type=['zip'], key="upload_zip")
        if st.button("Upload Images", key="upload_images_button") and uploaded_files:
            uploaded_count = load_images_to_db(uploaded_files, folder_choice, base_name)
            st.success(f"{uploaded_count} image(s) uploaded to '{folder_choice}'!")
        if st.button("Upload ZIP", key="upload_zip_button") and uploaded_zip:
            uploaded_count = extract_zip(uploaded_zip, folder_choice)
            st.success(f"{uploaded_count} image(s) extracted to '{folder_choice}'!")

# -------------------------------
# CSS Styling
# -------------------------------
st.markdown("""
<style>
.folder-card {background: #f9f9f9; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
.folder-header {font-size:1.5em; color:#333; margin-bottom:10px;}
.image-grid {display:flex; flex-wrap:wrap; gap:10px;}
img {border-radius:4px; max-width:100px; object-fit:cover;}
.image-info {margin-top:10px; word-break:break-all; font-size:0.9em;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Main App UI
# -------------------------------
st.title("📸 Image Manager & Editor")

data = load_folders()
if st.session_state.zoom_folder is None:
    # Grid View
    if not data:
        st.info("No folders available. Admins can create folders in the sidebar.")
    for f in data:
        st.markdown(
            f'<div class="folder-card"><div class="folder-header">'
            f'{f["name"]} ({f["description"] or "No description"})</div>',
            unsafe_allow_html=True
        )
        images = get_images(f["folder"])
        if images:
            # ZIP Download
            base_names = list(set(img["base_name"] for img in images if img["base_name"]))
            if base_names:
                zip_base_name = st.selectbox(
                    f"Select base name for ZIP ({f['name']})",
                    base_names,
                    key=f"zip_base_name_{f['folder']}"
                )
                if st.button(f"Download ZIP ({f['name']})", key=f"zip_download_{f['folder']}"):
                    zip_buffer = create_zip(f["folder"], zip_base_name)
                    st.download_button(
                        label=f"Download {zip_base_name}_images.zip",
                        data=zip_buffer,
                        file_name=f"{zip_base_name}_images.zip",
                        mime="application/zip",
                        key=f"zip_download_button_{f['folder']}"
                    )
            # Image Grid
            cols = st.columns(4)
            for idx, img_dict in enumerate(images):
                with cols[idx % 4]:
                    if st.button("🔍 Edit", key=f"edit_{f['folder']}_{idx}"):
                        st.session_state.zoom_folder = f["folder"]
                        st.session_state.zoom_index = idx
                        st.session_state.zoom_level = 1.0
                        st.session_state.zoom_center_x = 0.5
                        st.session_state.zoom_center_y = 0.5
                        st.session_state.crop_box = None
                        st.session_state.rotation = 0.0
                        st.session_state.brightness = 1.0
                        st.session_state.contrast = 1.0
                        st.session_state.preview_image = None
                        st.session_state.crop_preview_image = None
                        st.rerun()
                    st.image(img_dict["image"], use_container_width=True, caption=f"Photo {idx+1}")
                    st.markdown(f'<div class="image-info"><b>Name:</b> {img_dict["name"]}<br><b>Label:</b> {img_dict["label"]}</div>',
                                unsafe_allow_html=True)
                    if st.session_state.is_admin:
                        if st.button("🗑️ Delete", key=f"delete_grid_{f['folder']}_{img_dict['name']}"):
                            delete_image(f["folder"], img_dict["name"])
                            st.success("Image deleted.")
                            st.rerun()
        else:
            st.warning(f"No images in '{f['name']}'")

else:
    # Zoom/Edit View
    folder = st.session_state.zoom_folder
    images = get_images(folder)
    idx = st.session_state.zoom_index
    if idx >= len(images):
        idx = 0
        st.session_state.zoom_index = 0
    img_dict = images[idx]

    st.subheader(f"🔍 Editing {folder} ({idx+1}/{len(images)})")

    # Image Editing Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        # Display Image (Preview or Original)
        if st.session_state.preview_image:
            st.image(st.session_state.preview_image, use_container_width=True, caption="Preview")
        else:
            st.image(img_dict["image"], use_container_width=True, caption="Original Image")
        # Display Crop Preview if available
        if st.session_state.crop_preview_image:
            st.image(st.session_state.crop_preview_image, use_container_width=True, caption="Crop Preview with Borders")
    with col2:
        # Zoom Controls
        st.subheader("Zoom")
        zoom_level = st.slider("Zoom Level", 1.0, 5.0, st.session_state.zoom_level, key=f"zoom_{folder}_{idx}")
        center_x = st.slider("Center X", 0.0, 1.0, st.session_state.zoom_center_x, key=f"center_x_{folder}_{idx}")
        center_y = st.slider("Center Y", 0.0, 1.0, st.session_state.zoom_center_y, key=f"center_y_{folder}_{idx}")

        # Crop Controls
        st.subheader("Crop")
        st.markdown("Crop X and Crop Y are the coordinates starting at the top-left corner (0,0). Crop Width and Height are the distances (in pixels) relative to these X and Y coordinates.")
        crop_x = st.slider("Crop X", 0, img_dict["image"].width, 0, key=f"crop_x_{folder}_{idx}")
        crop_y = st.slider("Crop Y", 0, img_dict["image"].height, 0, key=f"crop_y_{folder}_{idx}")
        crop_w = st.slider("Crop Width", 1, img_dict["image"].width - crop_x, 100, key=f"crop_w_{folder}_{idx}")
        crop_h = st.slider("Crop Height", 1, img_dict["image"].height - crop_y, 100, key=f"crop_h_{folder}_{idx}")

        # Preview Crop Button
        if st.button("Preview Crop", key=f"preview_crop_{folder}_{idx}"):
            crop_box_temp = (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None
            st.session_state.crop_preview_image = draw_crop_overlay(img_dict["image"], crop_box_temp)
            st.rerun()

        # Rotation Control
        st.subheader("Rotation")
        rotation = st.slider("Rotation (degrees)", -180.0, 180.0, st.session_state.rotation, key=f"rotation_{folder}_{idx}")

        # Brightness and Contrast Controls
        st.subheader("Adjustments")
        brightness = st.slider("Brightness", 0.0, 2.0, st.session_state.brightness, key=f"brightness_{folder}_{idx}")
        contrast = st.slider("Contrast", 0.0, 2.0, st.session_state.contrast, key=f"contrast_{folder}_{idx}")

        # Preview Button
        if st.button("Preview", key=f"preview_{folder}_{idx}"):
            processed, _ = apply_transformations(
                img_dict["image"],
                zoom_level,
                center_x,
                center_y,
                (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None,
                rotation,
                brightness,
                contrast
            )
            st.session_state.preview_image = processed
            st.session_state.zoom_level = zoom_level
            st.session_state.zoom_center_x = center_x
            st.session_state.zoom_center_y = center_y
            st.session_state.crop_box = (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None
            st.session_state.rotation = rotation
            st.session_state.brightness = brightness
            st.session_state.contrast = contrast
            st.rerun()

        # Apply Changes Button
        if st.button("Apply Changes", key=f"apply_{folder}_{idx}"):
            processed, _ = apply_transformations(
                img_dict["image"],
                zoom_level,
                center_x,
                center_y,
                (crop_x, crop_y, crop_w, crop_h) if crop_w > 0 and crop_h > 0 else None,
                rotation,
                brightness,
                contrast
            )
            save_image_to_db(folder, img_dict["name"], processed)
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.success("Changes applied and saved to database.")
            st.rerun()

        # Reset Button
        if st.button("Reset", key=f"reset_{folder}_{idx}"):
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.rotation = 0.0
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.rerun()

        # Background Removal
        if st.button("Remove Background", key=f"remove_bg_{folder}_{idx}"):
            processed = remove_background(img_dict["image"])
            save_image_to_db(folder, img_dict["name"], processed)
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.success("Background removed and saved.")
            st.rerun()

        # Mole Removal
        if st.button("Remove Mole", key=f"remove_mole_{folder}_{idx}"):
            if st.session_state.crop_box is None:
                st.warning("Please set a crop box around the mole area first.")
            else:
                processed = inpaint_image(
                    img_dict["image"],
                    st.session_state.crop_box,
                    "smooth clear skin without mole or blemish, high quality"
                )
                save_image_to_db(folder, img_dict["name"], processed)
                st.session_state.preview_image = None
                st.session_state.crop_preview_image = None
                st.success("Mole removed using AI and saved.")
                st.rerun()

        # Cloth Removal
        if st.button("Remove Cloth", key=f"remove_cloth_{folder}_{idx}"):
            if st.session_state.crop_box is None:
                st.warning("Please set a crop box around the cloth area first.")
            else:
                processed = inpaint_image(
                    img_dict["image"],
                    st.session_state.crop_box,
                    "realistic nude human body, high quality, detailed skin"
                )
                save_image_to_db(folder, img_dict["name"], processed)
                st.session_state.preview_image = None
                st.session_state.crop_preview_image = None
                st.success("Cloth removed using AI and saved.")
                st.rerun()

    # Label and Classify
    label = st.text_input("Label", value=img_dict["label"], key=f"label_{folder}_{idx}")
    if st.button("Update Label", key=f"label_button_{folder}_{idx}"):
        update_image_label(folder, img_dict["name"], label)
        st.success("Label updated.")
        st.rerun()

    if st.session_state.is_admin:
        st.subheader("Classify Image")
        data = load_folders()
        new_folder = st.selectbox("Select Destination Folder", [item["folder"] for item in data], key=f"classify_folder_{folder}_{idx}")
        new_base_name = st.text_input("New Base Name", value="classified_image", key=f"classify_base_name_{folder}_{idx}")
        if st.button("Classify Image", key=f"classify_{folder}_{idx}"):
            classify_image(folder, img_dict["name"], new_folder, new_base_name)
            st.success(f"Image classified to '{new_folder}' as '{new_base_name}'.")
            st.session_state.zoom_index = max(0, idx-1)
            if len(get_images(folder)) == 0:
                st.session_state.zoom_folder = None
                st.session_state.zoom_index = 0
            st.rerun()

    # Download
    mime, _ = mimetypes.guess_type(img_dict["name"])
    st.download_button(
        "⬇️ Download",
        data=img_dict["data"],
        file_name=img_dict["name"],
        mime=mime,
        key=f"download_{folder}_{img_dict['name']}"
    )

    if st.session_state.is_admin:
        if st.button("🗑️ Delete Image", key=f"delete_zoom_{folder}_{img_dict['name']}"):
            delete_image(folder, img_dict["name"])
            st.success("Image deleted.")
            st.session_state.zoom_index = max(0, idx-1)
            if len(get_images(folder)) == 0:
                st.session_state.zoom_folder = None
                st.session_state.zoom_index = 0
            st.rerun()

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        if idx > 0 and st.button("◄ Previous", key=f"prev_{folder}_{idx}"):
            st.session_state.zoom_index -= 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.rotation = 0.0
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.rerun()
    with col3:
        if idx < len(images)-1 and st.button("Next ►", key=f"next_{folder}_{idx}"):
            st.session_state.zoom_index += 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.rotation = 0.0
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.preview_image = None
            st.session_state.crop_preview_image = None
            st.rerun()

    if st.button("⬅️ Back to Grid", key=f"back_{folder}_{idx}"):
        st.session_state.zoom_folder = None
        st.session_state.zoom_index = 0
        st.session_state.zoom_level = 1.0
        st.session_state.zoom_center_x = 0.5
        st.session_state.zoom_center_y = 0.5
        st.session_state.crop_box = None
        st.session_state.rotation = 0.0
        st.session_state.brightness = 1.0
        st.session_state.contrast = 1.0
        st.session_state.preview_image = None
        st.session_state.crop_preview_image = None
        st.rerun()
