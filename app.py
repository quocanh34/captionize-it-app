from flask import Flask, render_template, request
from inference import inference
import os
import dotenv

app = Flask(__name__)

config_env = dotenv.dotenv_values(".env")
app.config['UPLOAD_FOLDER'] = config_env["UPLOAD_FOLDER"]

def caption_image(image_path):
    captions = inference(image_path,
                         checkpoint_path=config_env["CHECKPOINT_PATH"],
                         device=config_env["DEVICE"],
                         vocab_path=config_env["VOCAB_PATH"]
                         )
    return captions, os.path.basename(image_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webm'}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'image' not in request.files:
            return render_template('index.html', error='No image file uploaded')

        file = request.files['image']

        # Check if the file has a valid extension
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if not allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type')

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)


        # Perform inference and get the captions
        captions, file_name = caption_image(file_path)

        # Render the captions and image on the page
        return render_template('index.html', captions=captions, file_name=file_name, image=file_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
