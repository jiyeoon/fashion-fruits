from flask import Flask
from flask import render_template, request, Response
from flask import url_for
from PIL import Image
from datetime import datetime
import os

from image_preprocessing import get_result_image

#app = Flask(__name__)
app = Flask(__name__, template_folder="./templates/", static_url_path="/static", static_folder="static")


@app.route('/')
def index():
    #return "Hello, World!"
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def get_result():
    if request.method == 'POST':
        width, height = 800, 800
        try:
            try:
                source = Image.open(request.files['source'])
            except:
                return Response("설마 여기서?!", status=400)
            print("1")
            now = datetime.now()
            dt = now.strftime("%m%d%H%M%S")
            try:
                source_img_path = os.path.dirname(os.path.abspath('__file__')) + '/static/inputs/source_' + dt + '.png' #'./static/inputs/source_' + dt + '.png'
                source.save(source_img_path)
            except:
                return Response('file load error. static/input/source 못찾음', status=400)
            print("2 ", source_img_path)
            #source.save(source_img_path)
            print("3 source image save success!")

            #input_img_path = './static/inputs/source.jpg'
            try:
                result_img_path = get_result_image(source_img_path)
            except:
                return Response("이미지 못불러옴. 처리는 함", status=400)
            # todo..
            # img 라는 변수에 classification 예스~!
            print("result_img_path : ", result_img_path)
        except Exception as e:
            print("Error!")
            return Response('fail', status=400)
    
    return result_img_path # 이미지 경로를 리턴하도록 해야함.. 


if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port='80', debug=True)