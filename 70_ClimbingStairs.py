from math import comb

n = 3

def climbStairs(n: int) -> int:
    result = 0
    for i in range(n//2+1):
        result += comb(n-i,i)
    return result


print(climbStairs(n))