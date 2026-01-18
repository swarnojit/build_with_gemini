# build_with_gemini
🌳 Afforestation Sapling Survival Analysis System

An AI-powered drone imagery analysis tool that monitors sapling survival in afforestation projects using Google Gemini 2.5 Flash.

📌 Overview

The system compares post-pitting (OP1) and post-plantation (OP3) drone orthomosaic images to automatically:

Detect planting pits

Identify surviving saplings

Calculate survival and mortality rates

Generate detailed reports

Built for large-scale afforestation monitoring with minimal manual effort.

✨ Key Features

🤖 AI-Based Image Analysis using Gemini 2.5 Flash

🛰️ Handles Very Large Drone Images (400M+ pixels)

🧩 Chunked Image Processing for accurate spatial analysis

📊 Survival & Mortality Metrics in real time

📁 Export Reports in JSON, CSV, and text formats

🖥️ Interactive Streamlit Dashboard

⚙️ Tech Stack

Python

Streamlit

Google Gemini API

Pillow, NumPy, Pandas

🚀 How It Works (Quick Flow)

Load OP1 (post-pitting) and OP3 (post-plantation) images

Compress & optionally split images into chunks

Gemini analyzes pits vs saplings

Results are aggregated

Survival statistics and reports are generated

📊 Outputs

Total pits detected

Surviving saplings

Casualties

Survival & mortality percentages

Confidence score based on analysis coverage

🧪 Use Cases

Government afforestation audits

NGO plantation monitoring

CSR environmental reporting

Smart forestry & climate projects

🔒 Security

API key stored securely using .env

No sensitive data committed to version control

🏁 Version

v1.0.0 – Initial Release

Large-image support

Chunk-based AI analysis

Automated reporting
