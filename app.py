from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('webpage.html')  # Your HTML file

@app.route('/start_eye_tracker', methods=['GET'])
def start_eye_tracker():
    try:
        subprocess.Popen(["python", "eye_tracker.py"])  # Run your Python script
        return jsonify({"message": "Eye tracker started successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
