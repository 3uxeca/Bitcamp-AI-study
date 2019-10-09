from flask import Flask

app = Flask(__name__)

@app.route("/<name>") # name은 아무거나 쓰면 그 문자가 출력된다 이 name이 밑에 user(name)에 들어가고 그게 다시 리턴값에 들어간다.
def user(name):
    return '<h1>Hello, %s !!!</h1>' %name

@app.route("/user/<name>")
def user2(name):
    return '<h1>Hello, user/%s !!!</h1>' %name

if __name__ == '__main__' :
    # app.run(host='127.0.0.1', port=5000, debug=False)
    app.run(host="192.168.0.129", port=8400, debug=False)
