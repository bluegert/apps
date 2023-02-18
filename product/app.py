from flask import Flask, render_template, request
from flask_login import login_required, current_user
import firebase_admin
from firebase_admin import credentials, storage

# Initialize the Flask app and Firebase SDK
app = Flask(__name__)
app.secret_key = '02fd1153eabbe9ccee9b89db'

cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred, {"storageBucket": "axonn-5a4a8.appspot.com"})
bucket = storage.bucket()

# Login page route
@app.route('/login')
def login():
    # Render the login template
    return render_template('login.html')

# Upload page route
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the uploaded file and its filename
        file = request.files['file']
        filename = file.filename
        # Upload the file to Firebase Storage
        blob = bucket.blob(current_user.id + '/' + filename)
        blob.upload_from_file(file)
        return 'File uploaded successfully!'
    # If the request method is GET, render the upload template
    return render_template('upload.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)