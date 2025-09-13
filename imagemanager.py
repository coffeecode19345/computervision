import unittest
import os
import io
import zipfile
import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sqlite3
from contextlib import contextmanager

# Configuration
APP_URL = "http://localhost:8501"
DB_PATH = "image_manager.db"
TEST_FOLDER = "testfolder"
TEST_NAME = "Test Folder"
TEST_BASE_NAME = "testimage"
ADMIN_PASSWORD = "admin123"  # Replace with st.secrets["admin_password"] in production
TEST_IMAGE_SIZE_MB = 0.5  # Size of each test image (MB)
MAX_SELECTED_SIZE_MB = 100

@contextmanager
def get_db_connection():
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def create_test_image(filename, size_mb=0.5):
    """Create a test JPEG image of approximately size_mb MB."""
    img = Image.new('RGB', (1000, 1000), color='blue')
    buffer = io.BytesIO()
    quality = 95
    img.save(buffer, format='JPEG', quality=quality)
    while buffer.tell() < size_mb * 1024 * 1024:
        img = Image.new('RGB', (int(img.width * 1.1), int(img.height * 1.1)), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
    with open(filename, 'wb') as f:
        f.write(buffer.getvalue())

def create_test_zip(filename, num_images=3):
    """Create a ZIP file with num_images test images."""
    with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i in range(num_images):
            img_path = f"temp_image_{i}.jpg"
            create_test_image(img_path, TEST_IMAGE_SIZE_MB)
            zip_file.write(img_path, f"test_{i}.jpg")
            os.remove(img_path)

class TestImageManager(unittest.TestCase):
    def setUp(self):
        """Set up Selenium WebDriver and clean database."""
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.driver.get(APP_URL)
        self.wait = WebDriverWait(self.driver, 10)
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM images WHERE folder = ?", (TEST_FOLDER,))
            c.execute("DELETE FROM folders WHERE folder = ?", (TEST_FOLDER,))
            c.execute("DELETE FROM image_history WHERE folder = ?", (TEST_FOLDER,))
            conn.commit()
        self.test_image = "test_image.jpg"
        self.test_zip = "test_images.zip"
        create_test_image(self.test_image, TEST_IMAGE_SIZE_MB)
        create_test_zip(self.test_zip, 3)

    def tearDown(self):
        """Clean up files and WebDriver."""
        self.driver.quit()
        if os.path.exists(self.test_image):
            os.remove(self.test_image)
        if os.path.exists(self.test_zip):
            os.remove(self.test_zip)

    def login(self):
        """Log in as admin."""
        self.driver.find_element(By.ID, "bui2").send_keys(ADMIN_PASSWORD)
        self.driver.find_element(By.ID, "bui3").click()
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Logged in as admin!')]")))

    def test_folder_creation(self):
        """Test creating a folder."""
        self.login()
        self.driver.find_element(By.ID, "bui4").send_keys(TEST_FOLDER)
        self.driver.find_element(By.ID, "bui5").send_keys(TEST_NAME)
        self.driver.find_element(By.ID, "bui6").send_keys("Test description")
        self.driver.find_element(By.ID, "bui7").click()
        self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), '{TEST_FOLDER}')]")))
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, description FROM folders WHERE folder = ?", (TEST_FOLDER,))
            result = c.fetchone()
            self.assertEqual(result, (TEST_NAME, "Test description"))

    def test_image_upload(self):
        """Test uploading a single image."""
        self.login()
        self.driver.find_element(By.ID, "bui8").send_keys(os.path.abspath(self.test_image))
        self.driver.find_element(By.ID, "bui9").send_keys(TEST_BASE_NAME)
        self.driver.find_element(By.ID, "bui10").click()
        self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), '1 image(s) uploaded')]")))
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, sequence FROM images WHERE folder = ? AND base_name = ?", (TEST_FOLDER, TEST_BASE_NAME))
            result = c.fetchone()
            self.assertEqual(result[0], f"{TEST_BASE_NAME}_1.jpg")
            self.assertEqual(result[1], 1)

    def test_zip_upload(self):
        """Test uploading a ZIP file."""
        self.login()
        self.driver.find_element(By.ID, "bui11").send_keys(os.path.abspath(self.test_zip))
        self.driver.find_element(By.ID, "bui9").send_keys(TEST_BASE_NAME)
        self.driver.find_element(By.ID, "bui12").click()
        self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), '3 image(s) extracted')]")))
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, sequence FROM images WHERE folder = ? AND base_name = ? ORDER BY sequence DESC", (TEST_FOLDER, TEST_BASE_NAME))
            results = c.fetchall()
            expected = [(f"{TEST_BASE_NAME}_{i}.jpg", i) for i in range(3, 0, -1)]
            self.assertEqual(results, expected)

    def test_image_selection_and_zip(self):
        """Test selecting images and downloading ZIP within 100 MB limit."""
        self.test_zip_upload()  # Upload 3 images
        checkboxes = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, "//input[@type='checkbox']")))
        for i in range(2):  # Select 2 images (~1 MB total)
            checkboxes[i].click()
        self.driver.find_element(By.ID, f"bui13_{TEST_FOLDER}").click()
        download_button = self.wait.until(EC.element_to_be_clickable((By.ID, f"bui14_{TEST_FOLDER}")))
        download_button.click()
        time.sleep(2)  # Wait for download
        zip_path = os.path.join(os.path.expanduser("~"), "Downloads", f"selected_{TEST_FOLDER}_images.zip")
        self.assertTrue(os.path.exists(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            self.assertEqual(len(zip_file.namelist()), 2)
        os.remove(zip_path)

    def test_crop_and_undo(self):
        """Test cropping an image and undoing the operation."""
        self.test_image_upload()
        self.driver.find_element(By.ID, f"bui15_{TEST_FOLDER}_0").click()  # Edit first image
        self.driver.find_element(By.ID, f"bui16_{TEST_FOLDER}_0").send_keys("100")  # Crop X
        self.driver.find_element(By.ID, f"bui17_{TEST_FOLDER}_0").send_keys("100")  # Crop Y
        self.driver.find_element(By.ID, f"bui18_{TEST_FOLDER}_0").send_keys("500")  # Crop Width
        self.driver.find_element(By.ID, f"bui19_{TEST_FOLDER}_0").send_keys("500")  # Crop Height
        self.driver.find_element(By.ID, f"bui20_{TEST_FOLDER}_0").click()  # Apply Crop
        self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), '{TEST_BASE_NAME}_1_p1.jpg')]")))
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, process_count FROM images WHERE folder = ? AND name = ?", (TEST_FOLDER, f"{TEST_BASE_NAME}_1_p1.jpg"))
            self.assertEqual(c.fetchone(), (f"{TEST_BASE_NAME}_1_p1.jpg", 1))
        self.driver.find_element(By.ID, f"bui21_{TEST_FOLDER}_0").click()  # Undo
        self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), '{TEST_BASE_NAME}_1.jpg')]")))
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, process_count FROM images WHERE folder = ? AND name = ?", (TEST_FOLDER, f"{TEST_BASE_NAME}_1.jpg"))
            self.assertEqual(c.fetchone(), (f"{TEST_BASE_NAME}_1.jpg", 0))

    def test_background_removal(self):
        """Test background removal and processed naming."""
        self.test_image_upload()
        self.driver.find_element(By.ID, f"bui15_{TEST_FOLDER}_0").click()  # Edit first image
        self.driver.find_element(By.ID, f"bui22_{TEST_FOLDER}_0").click()  # Remove Background
        self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), '{TEST_BASE_NAME}_1_p1.png')]")))
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name, process_count FROM images WHERE folder = ? AND name = ?", (TEST_FOLDER, f"{TEST_BASE_NAME}_1_p1.png"))
            self.assertEqual(c.fetchone(), (f"{TEST_BASE_NAME}_1_p1.png", 1))

    def test_label_styling(self):
        """Test label font size and placement."""
        self.test_image_upload()
        self.driver.find_element(By.ID, f"bui15_{TEST_FOLDER}_0").click()  # Edit first image
        label_input = self.wait.until(EC.presence_of_element_located((By.ID, f"bui23_{TEST_FOLDER}_0")))
        label_input.send_keys("Test Label")
        self.driver.find_element(By.ID, f"bui24_{TEST_FOLDER}_0").click()  # Update Label
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Label updated')]")))
        # Check grid view label styling
        self.driver.find_element(By.ID, f"bui25_{TEST_FOLDER}_0").click()  # Back to Grid
        label_element = self.wait.until(EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), 'Test Label')]")))
        style = label_element.value_of_css_property("font-size")
        self.assertEqual(style, "16px")  # 1.0em = 16px typically
        self.assertEqual(label_element.value_of_css_property("text-align"), "center")

if __name__ == "__main__":
    unittest.main()
