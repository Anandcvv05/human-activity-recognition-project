🎥 Human Action Recognition System
📌 Project Description

This project is a Human Action Recognition System that detects and classifies human actions from images/videos using Deep Learning.

It uses a trained model to analyze visual input and predict actions such as walking, running, sitting, etc.

🚀 Features
Upload image or video for prediction
Deep learning-based action detection
User-friendly web interface (Flask)
Real-time prediction support
Model training and prediction modules
🛠️ Technologies Used
Python
Flask
TensorFlow / Keras
OpenCV
NumPy
HTML, CSS

📂 Project Structure: -
human_action_recognition/
│
├── app.py                # Main Flask app
├── train.py             # Model training
├── predict.py           # Prediction logic
├── utils.py             # Helper functions
├── requirements.txt     # Dependencies
│
├── models/              # Trained models
├── dataset/             # Dataset
├── uploads/             # Uploaded files
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/human-action-recognition.git
cd human-action-recognition
2. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run the application
python app.py
🌐 Usage
Open browser
Go to: http://127.0.0.1:5000
Upload image/video
View predicted action
📊 Output
Displays predicted human action
Shows processed results
🎯 Future Improvements
Real-time webcam detection
Better model accuracy
More action classes
Deployment on cloud (Render / AWS)
👨‍💻 Author

CVV ANAND

B VIKAS

D SANDEEP

C GANESH

M VISHNU VARDHAN

CH YASHWANTH

📜 License

This project is for educational purposes.

✅ After creating file, run:
git add README.md
git commit -m "Added README file"
git push
