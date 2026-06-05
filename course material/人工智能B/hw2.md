# 12312515 王洛源

## Assignment 2

#### Q1：

Let the MDP be defined as $((S, A, P, R, \gamma))$ , where:

- $(S)$ :  finite set of states  
- $(A)$ :  finite set of actions  
- $(P(s'|s,a))$ :  transition probability  
- $(R(s,a))$ :  expected immediate reward  
- $(\gamma \in (0,1))$ : discount factor  

For a deterministic policy $\pi$, define:

$$
V^\pi(s) = \mathbb{E}_\pi \Big[ \sum_{t=0}^{\infty} \gamma^t r_t \,\Big|\, s_0=s \Big]
$$

and the state–action value function:

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$

We define a new policy $\pi'$ based on $\pi$ as:

$$
\pi'(s) = \arg\max_{a \in A} Q^\pi(s,a)
$$

that is, in each state, choose the action that maximizes $Q^\pi(s,a)$.

We need to prove that:

$$
V^{\pi'}(s) \ge V^{\pi}(s), \quad \forall s \in S
$$

From the Bellman equation for $V^\pi$:

$$
V^\pi(s) = Q^\pi(s,\pi(s))
$$

By the definition of $\pi'$:

$$
Q^\pi(s,\pi'(s)) \ge Q^\pi(s,\pi(s)) = V^\pi(s)
$$

Hence,

$$
R(s,\pi'(s)) + \gamma \sum_{s'} P(s'|s,\pi'(s)) V^\pi(s') \ge V^\pi(s)
$$

This can be rewritten as:

$$
T^{\pi'} V^\pi \ge V^\pi
$$

where $T^{\pi'}$ is the Bellman operator for policy $\pi'$.

Because $T^{\pi'}$ is a monotone contraction mapping, repeatedly applying it gives:

$$
V^{\pi'} = \lim_{k \to \infty} (T^{\pi'})^k V^\pi \ge V^\pi
$$

Thus, every policy improvement step produces a policy that performs no worse than the previous one.

Because the MDP has finite $S$ and $A$, the total number of possible deterministic policies is finite.  
Each improvement step satisfies:

$$
V^{\pi_{k+1}}(s) \ge V^{\pi_k}(s), \quad \forall s
$$

and if $V^{\pi_{k+1}} = V^{\pi_k}$, then $\pi_{k+1} = \pi_k$, i.e. the algorithm is optimal.

Therefore, the sequence of policies

$$
\pi_0, \pi_1, \pi_2, \dots
$$

is monotonic non-decreasing in value and must converge in a finite number of steps to some policy $\pi^*$ .

At convergence, $\pi^*$ satisfies:

$$
\pi^*(s) = \arg\max_a Q^{\pi^*}(s,a)
$$

which is precisely the optimal policy.

---

#### Q2：

We will show that $\{V_n\}$ is a Cauchy sequence. 

From the update rule,

$$
V_n - V_{n-1}
= (1-\alpha_n)V_{n-1} + \alpha_n x_n - V_{n-1}
= \alpha_n\,(x_n - V_{n-1}).2)

$$

Taking absolute values and using boundedness,

$$
|V_n - V_{n-1}|
= \alpha_n\,|x_n - V_{n-1}|
\le \alpha_n(|x_n| + |V_{n-1}|)
\le \alpha_n\,(C_1 + C_2)
$$

Because $\alpha_n = 1/n^2$,

$$
|V_n - V_{n-1}| \le \frac{C_1 + C_2}{n^2}
$$

For any integers $m>n$,

$$
|V_m - V_n|
\le \sum_{k=n+1}^{m} |V_k - V_{k-1}|
\le (C_1 + C_2)\sum_{k=n+1}^{m} \frac{1}{k^2}
$$

Since $\sum_{k=1}^{\infty} \frac{1}{k^2}$ converges, for any $\varepsilon>0$ there exists $N$ such that whenever $m,n>N$,

$$
\sum_{k=n+1}^{m} \frac{1}{k^2} \;<\; \frac{\varepsilon}{C_1 + C_2}
$$

Thus,

$$
|V_m - V_n| \;<\; \varepsilon
$$

so $\{V_n\}$ is Cauchy.
