class A:

    def __init__(self):
        self.a =1

def init():
    global b 
    b = A()

if __name__ == "__main__":
    init()
    b.a = 4
    print(b.a)
