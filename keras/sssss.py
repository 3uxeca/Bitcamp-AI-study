'''
사용자에게 초 단위의 시간을 입력 받아 
몇 시간, 몇 분, 몇 초인지를 계산하는 프로그램을 작성하시오.
ex) 2457초는 몇시간 몇분 몇초?
'''

input_sec = input("초를 입력해주세요: ")
input_sec = int(input_sec)

hours = input_sec // 3600           # 초를 3600으로 나눈 몫이 시간다.
minutes  = ( input_sec % 3600 ) // 60  # 초를 시간으로 변환한 나머지 초를 60으로 나눈 몫
seconds  = ( input_sec % 3600 ) % 60   # 초를 시간으로 변환한 나머지 초를 60으로 나눈 나머지

print("입력받은 {}초는 {}시간 {}분 {}초".format(input_sec, hours, minutes, seconds))
'''
hour = input_sec // 3600
min = input_sec % 3600
# sec =  
print("hour", hour)
print("min", min)
# print("sec", sec)
'''