from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333(): # 위의 app.route 와 연결된 함수
    return "<h1>hello world!</h1>"

@app.route('/ping', methods=['GET']) # GET, POST 등 
def ping(): # 위의 app.route 와 연결된 함수
    return "<h1>pong</h1>"    

if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=5000, debug=False)
    app.run(host="192.168.0.129", port=8400, debug=False)