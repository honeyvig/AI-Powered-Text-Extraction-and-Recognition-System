# AI-Powered-Text-Extraction-and-Recognition-System
AI-Powered Text Extraction and Recognition System Development

Job Description:
Develop a robust offline OCR (Optical Character Recognition) solution for extracting and recognizing text from images. The goal is to create a lightweight, high-performing system that can accurately scan and extract key information (e.g., text on documents) and integrate seamlessly into a mobile application.
-----
To develop an AI-Powered Text Extraction and Recognition System using OCR (Optical Character Recognition), you can use popular Python libraries like Tesseract OCR (via the pytesseract wrapper) and OpenCV for image processing. Below is a Python code to build a basic OCR system that extracts text from images offline, which can then be integrated into a mobile application.
Steps to Implement the OCR System:

    Install Dependencies: You'll need to install the required libraries. Run the following commands to install them:

    pip install pytesseract opencv-python Pillow

    You also need to install Tesseract OCR on your system. Here are the installation instructions:
        Windows: Download and install Tesseract from here.
        Linux: Run sudo apt-get install tesseract-ocr.
        macOS: Run brew install tesseract.

    Python Code for OCR System:

import cv2
import pytesseract
from PIL import Image
import numpy as np

class OCRSystem:
    def __init__(self, tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"):
        """
        Initializes the OCR system by setting the path to the Tesseract executable.

        Args:
          tesseract_cmd (str): Path to the Tesseract executable (default is Windows path).
        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def preprocess_image(self, image_path):
        """
        Preprocess the image for OCR recognition.

        Args:
          image_path (str): Path to the image to process.

        Returns:
          Image: Processed image ready for OCR.
        """
        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform thresholding to make text stand out more clearly
        _, threshold_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

        # Use noise removal and dilation/erosion to improve text extraction
        kernel = np.ones((1, 1), np.uint8)
        img_dilated = cv2.dilate(threshold_img, kernel, iterations=1)

        return img_dilated

    def extract_text(self, preprocessed_img):
        """
        Extract text from the preprocessed image using Tesseract OCR.

        Args:
          preprocessed_img (Image): Preprocessed image for OCR.

        Returns:
          str: Extracted text from the image.
        """
        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(preprocessed_img)

        return extracted_text

    def save_extracted_text(self, extracted_text, output_file="extracted_text.txt"):
        """
        Save the extracted text to a file.

        Args:
          extracted_text (str): The text extracted from the image.
          output_file (str): The output file where the text will be saved.
        """
        with open(output_file, "w") as file:
            file.write(extracted_text)

    def ocr_from_image(self, image_path, output_file="extracted_text.txt"):
        """
        Perform OCR on an image and save the extracted text to a file.

        Args:
          image_path (str): Path to the image.
          output_file (str): Output text file.
        """
        preprocessed_img = self.preprocess_image(image_path)
        extracted_text = self.extract_text(preprocessed_img)
        self.save_extracted_text(extracted_text, output_file)
        return extracted_text

# Example Usage
if __name__ == "__main__":
    ocr_system = OCRSystem()

    # Path to your image (for example, a scanned document)
    image_path = "path_to_your_image.jpg"
    
    # Extract text and save it to a file
    extracted_text = ocr_system.ocr_from_image(image_path)

    print("Extracted Text:")
    print(extracted_text)

Breakdown of the Code:

    OCRSystem Class:
        The class OCRSystem encapsulates the key functionalities for preprocessing the image, extracting text via Tesseract, and saving the extracted text to a file.

    Initialization (__init__):
        The path to the Tesseract executable is set during initialization. The path provided is for a Windows system. You need to change the path based on your system's Tesseract installation location.

    Preprocess Image:
        Grayscale Conversion: The image is first converted to grayscale to simplify the extraction of text.
        Thresholding: This step enhances the contrast of the image, making the text clearer for OCR recognition.
        Noise Removal and Dilation: Dilation is applied to enhance the text's clarity, and any noise is removed.

    Extract Text:
        Tesseract OCR: Using the pytesseract library, we extract the text from the preprocessed image. This method can handle different types of text and fonts.

    Save Extracted Text:
        The extracted text is saved to a .txt file, which can be used for further processing or integration into a mobile application.

    OCR from Image:
        The main function ocr_from_image takes an image path, preprocesses the image, extracts the text, and then saves the result.

Example Usage:

In the example usage, you:

    Create an instance of the OCRSystem class.
    Provide the path to the image file containing the text you want to extract.
    The script will process the image and print the extracted text to the console.

Output:

If the image contains a scanned document with readable text, the extracted text will be printed in the console and saved to a file extracted_text.txt. This can be used for further processing or integrated into a mobile application.
Possible Improvements:

    Multilingual Support: You can extend Tesseract to support different languages by providing the language parameter (e.g., lang='eng+spa' for English and Spanish).
    Error Handling: Add more robust error handling for cases where the image is unreadable or contains no text.
    Advanced Image Preprocessing: Apply additional image processing techniques to improve OCR accuracy, such as edge detection, skew correction, or resizing.

Integrating with a Mobile Application:

Once the OCR system is working, you can integrate it into a mobile app using a mobile framework such as Kivy (for Python-based apps) or a native mobile development platform (Android/iOS). The core OCR functionality could be wrapped in a Python backend that the mobile app calls via API or directly included in the mobile app depending on the platform.
