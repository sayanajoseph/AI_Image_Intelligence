# AI Image Intelligence

An AI-powered computer vision application developed using Python and pretrained deep learning models. It allows users to upload images, classify their contents, and generate natural-language captions through an interactive web interface.

## Features

- **Image Classification:** Predicts the top five possible labels for an uploaded image.
- **Image Captioning:** Generates a text description of an uploaded image.
- **Interactive Web Interface:** Uses Streamlit for image uploads and displaying results.
- **Local Inference:** Uses pretrained models locally without requiring a paid AI API key.

## Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pillow

## Pretrained Models

| Task | Model |
|---|---|
| Image Classification | `google/vit-base-patch16-224` |
| Image Captioning | `Salesforce/blip-image-captioning-base` |

## How It Works

1. Upload an image.
2. Select image classification or image captioning.
3. The selected pretrained model processes the image.
4. View the predicted labels or generated caption.

## Installation and Usage

### Prerequisites

- Python 3.11
- Git
- `uv` package manager
- Internet connection for initial model downloads

### Setup

Clone the Repository
```bash
git clone https://github.com/sayanajoseph/AI_Image_Intelligence.git
cd AI_Image_Intelligence
```

Install dependencies:

```bash
uv sync
```

Run the application:

```bash
uv run streamlit run app.py
```

## Limitations

- Predictions and captions may not always be accurate.
- Initial model downloads require an internet connection.
- Processing speed depends on the computer's hardware.

## Future Improvements

- Improve caption quality and descriptive detail.
- Add example images and evaluation metrics.
- Improve the display of predictions and confidence scores.
- Explore additional computer vision tasks.

## Author

Sayana Joseph

AI Engineer Aspirant | BCA Graduate

Developed this project to apply computer vision and deep learning concepts using pretrained AI models.
