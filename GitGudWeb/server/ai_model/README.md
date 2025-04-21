# AI Model Integration for GitGudWeb

This directory contains the Python AI model for analyzing building damage from before and after images.

## Setup

1. Install Python 3.8+ if not already installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the AI Service

Start the AI service with:
```
python app.py
```

The service will run on port 5002 by default.

## API Endpoints

### POST /api/analyze
Analyzes before and after images to detect building damage.

**Request:**
- Form data with two files:
  - `beforeImage`: Image before damage
  - `afterImage`: Image after damage

**Response:**
```json
{
  "results": {
    "damagePercentage": 65,
    "severity": "Orta-Ağır",
    "recommendations": [
      "Kapsamlı yapısal değerlendirme gerekli",
      "Güçlendirme çalışması önerilir",
      "Kullanım kısıtlaması gerekebilir"
    ],
    "processedImages": {
      "difference": "path/to/difference/image.jpg",
      "highlighted": "path/to/highlighted/image.jpg"
    }
  }
}
```

## Integration with Node.js Server

The Node.js server communicates with this AI service to analyze images. The analysis route in the Node.js server forwards the image data to this service and returns the results to the client.

## Future Improvements

This is currently using a simple pixel difference algorithm as a placeholder. In the future, you can replace the `analyze_damage` function with a more sophisticated deep learning model for better damage assessment.
