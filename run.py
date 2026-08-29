import os
# Set GEMINI_API_KEY env var before importing app
# Either set it in your shell:  set GEMINI_API_KEY=your-key
# Or uncomment the line below:
# os.environ["GEMINI_API_KEY"] = "your-key-here"
from app import app
app.run(debug=False, port=3000)
