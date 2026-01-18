# 🌳 Afforestation Sapling Survival Analysis

AI-powered drone imagery analysis tool for monitoring afforestation projects using Google Gemini 2.5 Flash.

## 🚀 Quick Start

### Installation

```bash
pip install streamlit google-generativeai pillow python-dotenv pandas numpy
```

### Setup

1. Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open browser at `http://localhost:8501`

## 📋 Features

- **AI Analysis**: Automated sapling detection and survival rate calculation
- **Large Image Support**: Handles 400M+ pixel drone imagery
- **Smart Processing**: Image compression and chunked analysis
- **Export Options**: JSON, CSV, and text reports
- **Interactive Dashboard**: Real-time metrics and visualizations

## 🎯 Usage

1. **Configure** (Sidebar):
   - API key loads automatically from `.env`
   - Set image paths for OP1 (reference) and OP3 (current) images
   - Adjust compression settings (default: 3072px)
   - Configure patch information (area, saplings, spacing)

2. **Analyze**:
   - Click "🔍 Analyze Sapling Survival"
   - Wait for processing (2-5 minutes for large images)
   - Review results in dashboard

3. **Export**:
   - Download JSON report, CSV data, or text summary

## 📊 Key Metrics

- **Survival Rate**: Percentage of successfully growing saplings
- **Casualties**: Number of failed plantings
- **Confidence Level**: Analysis reliability (high/medium/low)

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Max Dimension | 3072px | Image compression size |
| JPEG Quality | 85 | Compression quality |
| Chunking | On | Split images for detailed analysis |
| Chunk Size | 1024px | Size of analysis chunks |

## 🔧 Troubleshooting

**API Key Error**: Add `GEMINI_API_KEY` to `.env` file  
**File Not Found**: Verify image paths in sidebar  
**Slow Processing**: Reduce max dimension or disable chunking  

## 📁 Project Structure

```
project/
├── app.py              # Main application
├── .env               # API key (do not commit!)
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## 🔒 Security

- Never commit `.env` to version control
- Add `.env` to `.gitignore`
- Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## 📝 License

Provided as-is for educational and research purposes.
