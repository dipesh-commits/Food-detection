from distutils.log import debug
from io import BytesIO
from PIL import Image
from app.segmentation import predict_mask
from flask import Flask, request, render_template, redirect
import pickle
import base64, uuid
import numpy as np

from segmentation import predict_mask

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        picture = request.files["file"]
        if picture.filename == '':
                print('No file selected')
                return redirect(request.url)
        if picture and allowed_file(picture.filename):
            org_filename = (picture.filename)
            img = Image.open(picture.stream)
            filename = uuid.uuid4().hex + ".jpg"
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                img.save(f"images/{filename}")
                image_bytes = buf.getvalue()
                encoded_string = base64.b64encode(image_bytes).decode()
                # predict mask
                binary_segmentation_mask = predict_mask(np.array(img))

            return render_template('index.html', img_data=encoded_string), 200
    else:
        return render_template('index.html', img_data=""), 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)

