from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333(): # 위의 app.route 와 연결된 함수
    return "<h1>hello 3uxe world!</h1>"

@app.route('/bit')
def hello444(): # 위의 app.route 와 연결된 함수
    return "<h1>hello bit computer world!</h1>"    

if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=5000, debug=False)
    app.run(host="192.168.0.129", port=8400, debug=False)

# 내 PC의 IP주소 127.0.0.1를 서버주소로 쓴다. / port는 임의의 수를 넣으면 된다 5000이상
# 192.168.0.129 - - [03/Sep/2019 10:07:38] "GET /bit/ HTTP/1.1" 200  <<< 여기서 200은 "정상적으로 서버가 잘 작동하고 있다"는 뜻
# 192.168.0.129 - - [03/Sep/2019 10:08:32] "GET /bitdd HTTP/1.1" 404 <<< 여기서 404는 "서버에서 해당 페이지를 찾을 수 없다"는 뜻(Not Found)

