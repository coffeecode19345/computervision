import streamlit as st
from PIL import Image
import io
import zipfile
import sqlite3
import uuid
import os
import base64
import mimetypes
from contextlib import contextmanager
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError as e:
    st.warning(f"Background removal is disabled: {str(e)}. Install 'rembg' for this feature.")
    REMBG_AVAILABLE = False

DB_PATH = "image_manager.db"
MAX_FILE_SIZE_MB = 2
MAX_SELECTED_SIZE_MB = 100
ALLOWED_TYPES = ['image/png', 'image/jpeg', '.png', '.jpg', '.jpeg']
THUMBNAIL_SIZE = (80, 80)
IMAGES_PER_PAGE = 20

# -------------------------------
# Helper Functions
# -------------------------------
@contextmanager
def get_db_connection():
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def image_to_base64(image_data):
    """Convert image data (bytes) to base64 string."""
    return base64.b64encode(image_data).decode('utf-8')

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

def compress_image(image_data, quality=85):
    """Compress image to reduce memory usage."""
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"Error compressing image: {str(e)}")
        return image_data

def extract_zip(zip_file, folder, base_name):
    """Extract images from a ZIP file to the database."""
    uploaded_count = 0
    try:
        zip_buffer = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (folder, base_name))
                max_sequence = c.fetchone()[0] or 0
                for idx, file_name in enumerate(zip_ref.namelist(), start=max_sequence + 1):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_data = zip_ref.read(file_name)
                        file_buffer = io.BytesIO(file_data)
                        file_buffer.name = file_name
                        if validate_file(file_buffer):
                            extension = os.path.splitext(file_name)[1].lower()
                            new_name = f"{base_name}_{idx}{extension}"
                            compressed_data = compress_image(file_data)
                            c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, new_name))
                            if c.fetchone()[0] == 0:
                                c.execute("INSERT INTO images (name, folder, image_data, base_name, sequence) VALUES (?, ?, ?, ?, ?)",
                                          (new_name, folder, compressed_data, base_name, idx))
                                uploaded_count += 1
                conn.commit()
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
    try:
        x, y, w, h = crop_box
        return img.crop((x, y, x+w, y+h))
    except Exception as e:
        st.error(f"Error cropping image: {str(e)}")
        return img

def remove_background(img):
    """Remove background from a PIL image using rembg with lightweight model."""
    if not REMBG_AVAILABLE or st.session_state.low_memory_mode:
        return img
    try:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        output = remove(img_bytes.read(), model_name='u2netp')
        return Image.open(io.BytesIO(output)).convert('RGBA')
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}")
        return img

def init_db():
    """Initialize SQLite database with folders, images, and history tables."""
    with get_db_connection() as conn:
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
                process_count INTEGER DEFAULT 0,
                FOREIGN KEY(folder) REFERENCES folders(folder)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS image_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                name TEXT NOT NULL,
                folder TEXT NOT NULL,
                image_data BLOB NOT NULL,
                operation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)
        default_folders = [
            {"name": "General", "description": "General images", "folder": "general"},
            {"name": "Classified", "description": "Classified images", "folder": "classified"},
        ]
        for folder_data in default_folders:
            c.execute("SELECT COUNT(*) FROM folders WHERE folder = ?", (folder_data["folder"],))
            if c.fetchone()[0] == 0:
                c.execute("INSERT INTO folders (folder, name, description) VALUES (?, ?, ?)",
                          (folder_data["folder"], folder_data["name"], folder_data["description"]))
        conn.commit()

def load_folders():
    """Load all folders from the database."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT folder, name, description FROM folders")
        return [{"folder": r[0], "name": r[1], "description": r[2]} for r in c.fetchall()]

def add_folder(folder, name, description):
    """Add a new folder to the database."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO folders (folder, name, description) VALUES (?, ?, ?)",
                      (folder, name, description or ""))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        st.error(f"Folder '{folder}' already exists.")
        return False
    except Exception as e:
        st.error(f"Error adding folder: {str(e)}")
        return False

def load_images_to_db(uploaded_files, folder, base_name):
    """Load uploaded images to the database with sequential naming."""
    uploaded_count = 0
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (folder, base_name))
        max_sequence = c.fetchone()[0] or 0
        for idx, uploaded_file in enumerate(uploaded_files, start=max_sequence + 1):
            if validate_file(uploaded_file):
                image_data = uploaded_file.read()
                extension = os.path.splitext(uploaded_file.name)[1].lower()
                new_name = f"{base_name}_{idx}{extension}"
                compressed_data = compress_image(image_data)
                c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, new_name))
                if c.fetchone()[0] == 0:
                    c.execute("INSERT INTO images (name, folder, image_data, base_name, sequence) VALUES (?, ?, ?, ?, ?)",
                              (new_name, folder, compressed_data, base_name, idx))
                    uploaded_count += 1
        conn.commit()
    return uploaded_count

def get_images(folder, page=1):
    """Retrieve images for a folder with pagination, sorted by sequence descending."""
    offset = (page - 1) * IMAGES_PER_PAGE
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name, image_data, label, base_name, sequence, process_count FROM images WHERE folder = ? ORDER BY sequence DESC LIMIT ? OFFSET ?",
                  (folder, IMAGES_PER_PAGE, offset))
        images = []
        for r in c.fetchall():
            img_id, name, data, label, base_name, sequence, process_count = r
            try:
                img = Image.open(io.BytesIO(data))
                img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
                base64_image = image_to_base64(data)
                images.append({
                    "id": img_id,
                    "name": name,
                    "image": img,
                    "data": data,
                    "label": label or "",
                    "base_name": base_name or "",
                    "sequence": sequence,
                    "process_count": process_count,
                    "base64": base64_image
                })
            except Exception as e:
                st.error(f"Error loading image {name}: {str(e)}")
        return images

def update_image_label(folder, name, label):
    """Update the label for an image."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("UPDATE images SET label = ? WHERE folder = ? AND name = ?", (label, folder, name))
        conn.commit()

def classify_image(folder, name, new_folder, new_base_name):
    """Move an image to a new folder with a new base name."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, image_data, label, sequence FROM images WHERE folder = ? AND name = ?", (folder, name))
        result = c.fetchone()
        if result:
            img_id, image_data, label, sequence = result
            c.execute("SELECT MAX(sequence) FROM images WHERE folder = ? AND base_name = ?", (new_folder, new_base_name))
            max_sequence = c.fetchone()[0] or 0
            new_sequence = max_sequence + 1
            extension = os.path.splitext(name)[1].lower()
            new_name = f"{new_base_name}_{new_sequence}{extension}"
            c.execute("INSERT INTO images (name, folder, image_data, label, base_name, sequence) VALUES (?, ?, ?, ?, ?, ?)",
                      (new_name, new_folder, image_data, label, new_base_name, new_sequence))
            c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
            c.execute("DELETE FROM image_history WHERE image_id = ?", (img_id,))
            conn.commit()

def delete_image(folder, name):
    """Delete an image and its history from the database."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM images WHERE folder = ? AND name = ?", (folder, name))
        result = c.fetchone()
        if result:
            img_id = result[0]
            c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
            c.execute("DELETE FROM image_history WHERE image_id = ?", (img_id,))
            conn.commit()

def save_image_history(image_id, name, folder, image_data, operation):
    """Save image state to history before modification."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO image_history (image_id, name, folder, image_data, operation) VALUES (?, ?, ?, ?, ?)",
                  (image_id, name, folder, image_data, operation))
        conn.commit()

def undo_last_operation(folder, name):
    """Undo the last operation for an image."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM images WHERE folder = ? AND name = ?", (folder, name))
        result = c.fetchone()
        if not result:
            return False
        img_id = result[0]
        c.execute("SELECT image_data, name FROM image_history WHERE image_id = ? ORDER BY timestamp DESC LIMIT 1", (img_id,))
        history = c.fetchone()
        if history:
            image_data, old_name = history
            c.execute("UPDATE images SET image_data = ?, name = ?, process_count = process_count - 1 WHERE id = ?",
                      (image_data, old_name, img_id))
            c.execute("DELETE FROM image_history WHERE image_id = ? AND image_data = ?", (img_id, image_data))
            conn.commit()
            return True
        return False

def create_zip_selected(selected_images):
    """Create a ZIP file for selected images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for img_dict in selected_images:
            zip_file.writestr(img_dict["name"], img_dict["data"])
    zip_buffer.seek(0)
    return zip_buffer

def create_zip(folder, base_name):
    """Create a ZIP file for all images in a folder with the given base name."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, image_data FROM images WHERE folder = ? AND base_name = ?", (folder, base_name))
            for name, data in c.fetchall():
                zip_file.writestr(name, data)
    zip_buffer.seek(0)
    return zip_buffer

# -------------------------------
# Initialize Session State
# -------------------------------
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "zoom_folder": None,
        "zoom_index": 0,
        "is_admin": False,
        "zoom_level": 1.0,
        "zoom_center_x": 0.5,
        "zoom_center_y": 0.5,
        "crop_box": None,
        "task_running": False,
        "low_memory_mode": False,
        "page": {},
        "selected_images": {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
init_db()

# -------------------------------
# Sidebar: Admin Controls & Upload
# -------------------------------
with st.sidebar:
    st.title("Image Manager Settings")
    st.session_state.low_memory_mode = st.checkbox("Low Memory Mode (Disables Background Removal)", value=st.session_state.low_memory_mode, key="low_memory_mode")
    
    st.title("Admin Login")
    with st.form(key="login_form"):
        pwd = st.text_input("Password", type="password", key="login_password")
        submit = st.form_submit_button("Login", disabled=st.session_state.task_running)
        if submit and pwd == "admin123":
            st.session_state.is_admin = True
            st.success("Logged in as admin!")
        elif submit:
            st.error("Incorrect password")
    if st.session_state.is_admin and st.button("Logout", key="logout_button", disabled=st.session_state.task_running):
        st.session_state.is_admin = False
        st.success("Logged out")
        st.rerun()

    if st.session_state.is_admin:
        st.subheader("Manage Folders & Images")
        with st.form(key="add_folder_form"):
            new_folder = st.text_input("Folder Name (e.g., 'newfolder')", key="new_folder_input")
            new_name = st.text_input("Display Name", key="new_name_input")
            new_description = st.text_area("Description (optional)", key="new_description_input")
            submit = st.form_submit_button("Add Folder", disabled=st.session_state.task_running)
            if submit and new_folder and new_name:
                with st.spinner("Adding folder..."):
                    st.session_state.task_running = True
                    if add_folder(new_folder.lower(), new_name, new_description):
                        st.success(f"Folder '{new_folder}' added successfully!")
                        st.rerun()
                    st.session_state.task_running = False

        st.subheader("Upload Images or ZIP")
        data = load_folders()
        folder_choice = st.selectbox("Select Folder", [item["folder"] for item in data], key="upload_folder_select")
        base_name = st.text_input("Enter base name for files:", value="image", key="base_name_input")
        uploaded_files = st.file_uploader(
            "Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload_files",
            disabled=st.session_state.task_running
        )
        uploaded_zip = st.file_uploader("Upload ZIP", type=['zip'], key="upload_zip", disabled=st.session_state.task_running)
        if st.button("Upload Images", key="upload_images_button", disabled=st.session_state.task_running or not uploaded_files):
            with st.spinner("Uploading images..."):
                st.session_state.task_running = True
                uploaded_count = load_images_to_db(uploaded_files, folder_choice, base_name)
                st.success(f"{uploaded_count} image(s) uploaded to '{folder_choice}'!")
                st.session_state.task_running = False
                st.rerun()
        if st.button("Upload ZIP", key="upload_zip_button", disabled=st.session_state.task_running or not uploaded_zip):
            with st.spinner("Extracting ZIP..."):
                st.session_state.task_running = True
                uploaded_count = extract_zip(uploaded_zip, folder_choice, base_name)
                st.success(f"{uploaded_count} image(s) extracted to '{folder_choice}'!")
                st.session_state.task_running = False
                st.rerun()

# -------------------------------
# CSS Styling
# -------------------------------
st.markdown("""
<style>
.folder-card {background: #f9f9f9; border-radius: 8px; padding: 10px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
.folder-header {font-size: 1.4em; color: #333; margin-bottom: 8px;}
.image-grid {display: flex; flex-wrap: wrap; gap: 8px;}
img {border-radius: 4px; max-width: 80px; object-fit: cover;}
.image-info {margin-top: 8px; word-break: break-all; font-size: 1.0em; line-height: 1.4; text-align: center;}
.image-info b {font-weight: 600;}
.stButton>button {width: 100%; font-size: 0.9em; margin-bottom: 5px;}
.label-input {font-size: 1.2em !important; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Main App UI
# -------------------------------
st.title("📸 Image Manager & Editor")

data = load_folders()
if not data:
    st.info("No folders available. Admins can create folders in the sidebar.")

if st.session_state.zoom_folder is None:
    # Grid View
    for f in data:
        st.markdown(
            f'<div class="folder-card"><div class="folder-header">'
            f'{f["name"]} ({f["description"] or "No description"})</div>',
            unsafe_allow_html=True
        )
        page = st.session_state.page.get(f["folder"], 1)
        images = get_images(f["folder"], page)
        if images:
            # Pagination
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM images WHERE folder = ?", (f["folder"],))
                total_images = c.fetchone()[0]
            total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
            col1, col2 = st.columns([1, 1])
            with col1:
                if page > 1 and st.button("Previous Page", key=f"prev_page_{f['folder']}", disabled=st.session_state.task_running):
                    st.session_state.page[f["folder"]] = page - 1
                    st.rerun()
            with col2:
                if page < total_pages and st.button("Next Page", key=f"next_page_{f['folder']}", disabled=st.session_state.task_running):
                    st.session_state.page[f["folder"]] = page + 1
                    st.rerun()
            st.write(f"Page {page} of {total_pages}")

            # ZIP Download for All Images
            base_names = list(set(img["base_name"] for img in images if img["base_name"]))
            if base_names:
                zip_base_name = st.selectbox(
                    f"Select base name for ZIP ({f['name']})",
                    base_names,
                    key=f"zip_base_name_{f['folder']}"
                )
                if st.button(f"Download ZIP ({f['name']})", key=f"zip_download_{f['folder']}", disabled=st.session_state.task_running):
                    with st.spinner("Creating ZIP..."):
                        st.session_state.task_running = True
                        zip_buffer = create_zip(f["folder"], zip_base_name)
                        st.download_button(
                            label=f"Download {zip_base_name}_images.zip",
                            data=zip_buffer,
                            file_name=f"{zip_base_name}_images.zip",
                            mime="application/zip",
                            key=f"zip_download_button_{f['folder']}"
                        )
                        st.session_state.task_running = False

            # Image Selection and ZIP
            if f["folder"] not in st.session_state.selected_images:
                st.session_state.selected_images[f["folder"]] = []
            selected_images = st.session_state.selected_images[f["folder"]]
            total_size = sum(len(img["data"]) for img in selected_images)
            if total_size > MAX_SELECTED_SIZE_MB * 1024 * 1024:
                st.warning(f"Selected images exceed {MAX_SELECTED_SIZE_MB} MB. Deselect some images.")
            if selected_images and total_size <= MAX_SELECTED_SIZE_MB * 1024 * 1024:
                if st.button(f"ZIP Selected Images ({len(selected_images)})", key=f"zip_selected_{f['folder']}", disabled=st.session_state.task_running):
                    with st.spinner("Creating ZIP for selected images..."):
                        st.session_state.task_running = True
                        zip_buffer = create_zip_selected(selected_images)
                        st.download_button(
                            label="Download Selected Images ZIP",
                            data=zip_buffer,
                            file_name=f"selected_{f['folder']}_images.zip",
                            mime="application/zip",
                            key=f"zip_selected_download_{f['folder']}"
                        )
                        st.session_state.task_running = False

            # Image Grid
            cols = st.columns(4)
            for idx, img_dict in enumerate(images):
                with cols[idx % 4]:
                    if st.checkbox("Select", key=f"select_{f['folder']}_{img_dict['name']}", disabled=st.session_state.task_running):
                        if img_dict not in selected_images and total_size + len(img_dict["data"]) <= MAX_SELECTED_SIZE_MB * 1024 * 1024:
                            selected_images.append(img_dict)
                        elif img_dict in selected_images:
                            selected_images.remove(img_dict)
                        st.session_state.selected_images[f["folder"]] = selected_images
                    if st.button("🔍 Edit", key=f"edit_{f['folder']}_{idx}", disabled=st.session_state.task_running):
                        st.session_state.zoom_folder = f["folder"]
                        st.session_state.zoom_index = idx
                        st.session_state.zoom_level = 1.0
                        st.session_state.zoom_center_x = 0.5
                        st.session_state.zoom_center_y = 0.5
                        st.session_state.crop_box = None
                        st.rerun()
                    st.image(img_dict["image"], use_container_width=True, caption=f"Photo {idx+1}")
                    st.markdown(f'<div class="image-info"><b>Name:</b> {img_dict["name"]}<br><b>Label:</b> {img_dict["label"]}</div>',
                                unsafe_allow_html=True)
                    if st.session_state.is_admin:
                        if st.button("🗑️ Delete", key=f"delete_grid_{f['folder']}_{img_dict['name']}", disabled=st.session_state.task_running):
                            with st.spinner("Deleting image..."):
                                st.session_state.task_running = True
                                delete_image(f["folder"], img_dict["name"])
                                if img_dict in selected_images:
                                    selected_images.remove(img_dict)
                                    st.session_state.selected_images[f["folder"]] = selected_images
                                st.success("Image deleted.")
                                st.session_state.task_running = False
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
        processed_image = img_dict["image"].copy()
        if st.session_state.crop_box:
            processed_image = crop_image(processed_image, st.session_state.crop_box)
        else:
            processed_image, _ = zoom_image(
                processed_image,
                st.session_state.zoom_level,
                st.session_state.zoom_center_x,
                st.session_state.zoom_center_y
            )
        st.image(processed_image, use_container_width=True, caption="Processed Image")
    with col2:
        zoom_level = st.slider("Zoom Level", 1.0, 3.0, st.session_state.zoom_level, key=f"zoom_{folder}_{idx}")
        center_x = st.slider("Center X", 0.0, 1.0, st.session_state.zoom_center_x, key=f"center_x_{folder}_{idx}")
        center_y = st.slider("Center Y", 0.0, 1.0, st.session_state.zoom_center_y, key=f"center_y_{folder}_{idx}")
        if st.button("Apply Zoom", key=f"apply_zoom_{folder}_{idx}", disabled=st.session_state.task_running):
            with st.spinner("Applying zoom..."):
                st.session_state.task_running = True
                st.session_state.zoom_level = zoom_level
                st.session_state.zoom_center_x = center_x
                st.session_state.zoom_center_y = center_y
                st.session_state.crop_box = None
                st.session_state.task_running = False
                st.rerun()

        crop_x = st.number_input("Crop X", 0, img_dict["image"].width, 0, key=f"crop_x_{folder}_{idx}")
        crop_y = st.number_input("Crop Y", 0, img_dict["image"].height, 0, key=f"crop_y_{folder}_{idx}")
        crop_w = st.number_input("Crop Width", 1, img_dict["image"].width - crop_x, min(100, img_dict["image"].width), key=f"crop_w_{folder}_{idx}")
        crop_h = st.number_input("Crop Height", 1, img_dict["image"].height - crop_y, min(100, img_dict["image"].height), key=f"crop_h_{folder}_{idx}")
        if st.button("Apply Crop", key=f"crop_{folder}_{idx}", disabled=st.session_state.task_running):
            with st.spinner("Applying crop..."):
                st.session_state.task_running = True
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT id, image_data, base_name, sequence, process_count FROM images WHERE folder = ? AND name = ?",
                              (folder, img_dict["name"]))
                    result = c.fetchone()
                    if result:
                        img_id, old_data, base_name, sequence, process_count = result
                        save_image_history(img_id, img_dict["name"], folder, old_data, "crop")
                        processed_image = crop_image(img_dict["image"], (crop_x, crop_y, crop_w, crop_h))
                        img_bytes = io.BytesIO()
                        processed_image.save(img_bytes, format='JPEG', quality=85, optimize=True)
                        img_bytes.seek(0)
                        new_name = f"{base_name}_{sequence}_p{process_count + 1}.jpg"
                        c.execute("UPDATE images SET image_data = ?, name = ?, process_count = ? WHERE id = ?",
                                  (img_bytes.read(), new_name, process_count + 1, img_id))
                        conn.commit()
                st.session_state.zoom_level = 1.0
                st.session_state.crop_box = None
                st.session_state.task_running = False
                st.rerun()

        if REMBG_AVAILABLE and not st.session_state.low_memory_mode:
            if st.button("Remove Background", key=f"remove_bg_{folder}_{idx}", disabled=st.session_state.task_running):
                with st.spinner("Removing background..."):
                    st.session_state.task_running = True
                    with get_db_connection() as conn:
                        c = conn.cursor()
                        c.execute("SELECT id, image_data, base_name, sequence, process_count FROM images WHERE folder = ? AND name = ?",
                                  (folder, img_dict["name"]))
                        result = c.fetchone()
                        if result:
                            img_id, old_data, base_name, sequence, process_count = result
                            save_image_history(img_id, img_dict["name"], folder, old_data, "background_removal")
                            processed_image = remove_background(img_dict["image"])
                            img_bytes = io.BytesIO()
                            processed_image.save(img_bytes, format='JPEG', quality=85, optimize=True)
                            img_bytes.seek(0)
                            new_name = f"{base_name}_{sequence}_p{process_count + 1}.jpg"
                            c.execute("UPDATE images SET image_data = ?, name = ?, process_count = ? WHERE id = ?",
                                      (img_bytes.read(), new_name, process_count + 1, img_id))
                            conn.commit()
                    st.session_state.task_running = False
                    st.rerun()

        if st.session_state.is_admin:
            if st.button("Undo Last Operation", key=f"undo_{folder}_{idx}", disabled=st.session_state.task_running):
                with st.spinner("Undoing operation..."):
                    st.session_state.task_running = True
                    if undo_last_operation(folder, img_dict["name"]):
                        st.success("Operation undone.")
                    else:
                        st.warning("No previous operation to undo.")
                    st.session_state.task_running = False
                    st.rerun()

    st.markdown('<div class="label-input">', unsafe_allow_html=True)
    label = st.text_input("Label", value=img_dict["label"], key=f"label_{folder}_{idx}")
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Update Label", key=f"label_button_{folder}_{idx}", disabled=st.session_state.task_running):
        with st.spinner("Updating label..."):
            st.session_state.task_running = True
            update_image_label(folder, img_dict["name"], label)
            st.success("Label updated.")
            st.session_state.task_running = False
            st.rerun()

    if st.session_state.is_admin:
        st.subheader("Classify Image")
        data = load_folders()
        new_folder = st.selectbox("Select Destination Folder", [item["folder"] for item in data], key=f"classify_folder_{folder}_{idx}")
        new_base_name = st.text_input("New Base Name", value="classified_image", key=f"classify_base_name_{folder}_{idx}")
        if st.button("Classify Image", key=f"classify_{folder}_{idx}", disabled=st.session_state.task_running):
            with st.spinner("Classifying image..."):
                st.session_state.task_running = True
                classify_image(folder, img_dict["name"], new_folder, new_base_name)
                st.success(f"Image classified to '{new_folder}' as '{new_base_name}'.")
                st.session_state.zoom_index = max(0, idx-1)
                if len(get_images(folder)) == 0:
                    st.session_state.zoom_folder = None
                    st.session_state.zoom_index = 0
                st.session_state.task_running = False
                st.rerun()

    mime = 'image/jpeg' if img_dict["name"].lower().endswith(('.jpg', '.jpeg')) else 'image/png'
    st.download_button(
        "⬇️ Download",
        data=img_dict["data"],
        file_name=img_dict["name"],
        mime=mime,
        key=f"download_{folder}_{img_dict['name']}",
        disabled=st.session_state.task_running
    )

    if st.session_state.is_admin:
        if st.button("🗑️ Delete Image", key=f"delete_zoom_{folder}_{img_dict['name']}", disabled=st.session_state.task_running):
            with st.spinner("Deleting image..."):
                st.session_state.task_running = True
                delete_image(folder, img_dict["name"])
                st.success("Image deleted.")
                st.session_state.zoom_index = max(0, idx-1)
                if len(get_images(folder)) == 0:
                    st.session_state.zoom_folder = None
                    st.session_state.zoom_index = 0
                st.session_state.task_running = False
                st.rerun()

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        if idx > 0 and st.button("◄ Previous", key=f"prev_{folder}_{idx}", disabled=st.session_state.task_running):
            st.session_state.task_running = True
            st.session_state.zoom_index -= 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.task_running = False
            st.rerun()
    with col3:
        if idx < len(images)-1 and st.button("Next ►", key=f"next_{folder}_{idx}", disabled=st.session_state.task_running):
            st.session_state.task_running = True
            st.session_state.zoom_index += 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.session_state.crop_box = None
            st.session_state.task_running = False
            st.rerun()

    if st.button("⬅️ Back to Grid", key=f"back_{folder}_{idx}", disabled=st.session_state.task_running):
        st.session_state.task_running = True
        st.session_state.zoom_folder = None
        st.session_state.zoom_index = 0
        st.session_state.zoom_level = 1.0
        st.session_state.zoom_center_x = 0.5
        st.session_state.zoom_center_y = 0.5
        st.session_state.crop_box = None
        st.session_state.task_running = False
        st.rerun()
