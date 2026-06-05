def countValidParentheses(s: str) -> int:
    MOD = 10**9 + 7
    n = len(s)
    dp = [0] * (n // 2 + 2)
    dp[0] = 1
    
    for i in range(n):
        new_dp = [0] * (n // 2 + 2)
        for j in range(n // 2 + 1):
            if dp[j] == 0:
                continue
            # 放 '('
            if s[i] == '(' or s[i] == '?':
                if j + 1 <= n // 2:
                    new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % MOD
            # 放 ')'
            if s[i] == ')' or s[i] == '?':
                if j > 0:
                    new_dp[j - 1] = (new_dp[j - 1] + dp[j]) % MOD
        dp = new_dp
    
    return dp[0]


if __name__ == "__main__":
    s = input().strip()
    print(countValidParentheses(s))
