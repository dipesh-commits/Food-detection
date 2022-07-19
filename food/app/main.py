from io import BytesIO
from PIL import Image
from segmentation import mask, predict_bbox
from flask import Flask, request, render_template, redirect
import cv2
import base64, uuid
import numpy as np

from segmentation import predict, mask
from loguru import logger

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
                img.save(f"static/{filename}")
                image_bytes = buf.getvalue()
                encoded_string = base64.b64encode(image_bytes).decode()

                # predict mask
                output = predict(np.array(img))
                segmented_mask = mask(img, output)
                segmented_filename = uuid.uuid4().hex + '.png'
                segmented_mask.save(f"static/{segmented_filename}")
                logger.info(f"Segmentation done..")
                detected_img, detected_results = predict_bbox(np.array(img), output)
                detected_filename = uuid.uuid4().hex + '.jpg'
                logger.info(f"Detection done....")
                cv2.imwrite(f"static/{detected_filename}", detected_img)

                # predictions = {
                #     "img_data": encoded_string,
                #     "segmented_img": segmented_mask
                # }

            return render_template('index.html', img_data = encoded_string, segmented = segmented_filename, detected = detected_filename, results=detected_results), 200
    else:
        return render_template('index.html', img_data=""), 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)
