# Image to Image Generator

A Django web application for image-to-image generation using Hugging Face Diffusers.

## Features

- **Image-to-Image Generation**: Transform images using text prompts
- **Text-to-Image**: Generate images from text descriptions
- **Iterative Refinement**: Continue improving generated images across multiple iterations
- **Model Configuration**: Create and manage multiple model configurations
- **Session Management**: Track generation history with sessions
- **Multiple Models**: Support for SDXL, SD 1.5, SD 2.1, and InstructPix2Pix

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or Apple Silicon Mac (MPS) or CPU

## Installation

1. Clone the repository:
```bash
cd /path/to/image-to-image
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

5. Run migrations:
```bash
python manage.py migrate
```

6. (Optional) Pre-download models:
```bash
python manage.py download_models --model sd15  # Fastest to download
# Or download all models:
python manage.py download_models
```

7. Run the development server:
```bash
python manage.py runserver
```

8. Open http://localhost:8000 in your browser

## Usage

### Basic Generation

1. Go to the home page
2. Optionally upload a base image
3. Enter a text prompt describing your desired changes
4. Click "Generate"
5. Wait for the image to be generated

### Iterative Refinement

1. After generating an image, click "Iterate on this"
2. Check "Use previous output as input"
3. Enter a new prompt for further refinement
4. Click "Generate" again

### Model Configuration

1. Go to Settings
2. Create a new configuration with your preferred settings:
   - **Model**: Choose from SDXL, SD 1.5, SD 2.1, or InstructPix2Pix
   - **Strength**: How much to change the input image (0.0-1.0)
   - **Guidance Scale**: How closely to follow the prompt (1-20)
   - **Inference Steps**: Quality vs speed tradeoff (20-100)
   - **Negative Prompt**: What to avoid in generation
3. Activate the configuration to use it for next generation

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| SDXL Refiner 1.0 | High quality, detailed output | Final refinement |
| Stable Diffusion 1.5 | Fast, good quality | Quick iterations |
| Stable Diffusion 2.1 | Improved quality over 1.5 | General purpose |
| InstructPix2Pix | Edit with natural language | Specific edits |

## Project Structure

```
image-to-image/
├── config/                 # Django project settings
├── img2img_app/           # Main application
│   ├── services/          # Business logic (DiffusionEngine)
│   ├── management/        # Django commands
│   ├── models.py          # Database models
│   ├── views.py           # View handlers
│   └── urls.py            # URL routing
├── templates/             # HTML templates
├── static/                # CSS and JavaScript
├── media/                 # Uploaded and generated images
└── models_cache/          # Downloaded model files
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | (required for some models) |
| `DJANGO_SECRET_KEY` | Django secret key | (auto-generated) |
| `DEBUG` | Debug mode | True |
| `ALLOWED_HOSTS` | Allowed hosts | localhost,127.0.0.1 |

## Models 
⏺ Simpler. Here are the commands using cache:                                                                                       
                                                                                                                                    
  # SDXL Base                                                                                                                       
  hf download stabilityai/stable-diffusion-xl-base-1.0 --include='*.json' --include='*.txt' --include='*.safetensors'               
  --exclude='*.bin' --exclude='*.fp16.*'                                                                                            
                                                                                                                                    
  # SDXL Inpainting                                                                                                                 
  hf download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 --include='*.json' --include='*.txt' --include='*.safetensors'       
  --exclude='*.bin' --exclude='*.fp16.*'                                                                                            
                                                                                                                                    
  # IP-Adapter                                                                                                                      
  hf download h94/IP-Adapter --include='sdxl_models/*' --include='models/*' --include='*.json' --exclude='*.bin'                    
                                                                                                                                    
  Cache location: ~/.cache/huggingface/hub/      


## Tips

- First generation will download the model (~2-7GB depending on model)
- Use CUDA GPU for 10-50x faster generation
- Lower inference steps (20-30) for faster previews
- Higher guidance scale (10-15) for more prompt adherence
- Use negative prompts to avoid unwanted artifacts
