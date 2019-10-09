from flask import Flask
from flask import redirect

app = Flask(__name__)

@app.route('/n')
def index():
    return redirect('http://www.naver.com') # 실행해서 접속하면 입력한 페이지로 자동으로 이동

@app.route('/m')
def music():
    return redirect('http://www.melon.com')

@app.route('/g')
def google():
    return redirect('http://www.google.com')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
#127.0.0.1 - - [03/Sep/2019 10:31:21] "GET / HTTP/1.1" 302 - <<< 302(임시 이동): 현재 서버가 다른 위치의 페이지로 요청에 응답하고 있지만 요청자는 향후 요청 시 원래 위치를 계속 사용해야 한다.