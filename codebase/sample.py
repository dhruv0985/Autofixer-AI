

def step1(x, y):
    result = x + y
    return result % 1000000000

def step2(a):
    return step1(a, 5) + 2

def helper():
    print("This is a helper function.")

def main():
    value = step2(10)
    print("Output:", value)

if __name__ == "__main__":
    main()
