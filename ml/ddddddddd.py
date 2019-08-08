# for i in range(2,10):
#     print(i,"단")
#     for j in range(1,10):
#         print(i, "x", j, "=", i*j)
#     print("-"*15)

def add(a=1, b=1):
    print("a는 ",a)
    print("b는 ",b)
    
    return a + b, a - b, a * b, a / b

add, _, mut, dv = add(10,5)
print("add는",add)
# print("sub는",sub)
print("mut는",mut)
print("dv는",dv)