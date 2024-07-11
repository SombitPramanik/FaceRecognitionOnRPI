from flask import Flask, send_file, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/GetImage')
def get_image():
    # Path to your image file
    image_path = 'tst.jpg'
    # Return the image file
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='192.168.1.110',debug=False)
