from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') # 내부 서버 페이지로 들어가기 위한 코드(리다이렉트와 비슷하나 외부가 아닌 내부서버!)

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)

if __name__ == "__main__" :
    app.run(host='127.0.0.1', port=5000, debug=False)