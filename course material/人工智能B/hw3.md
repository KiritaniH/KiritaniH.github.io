## 12312515 王洛源 第三次作业

---

### Q1

（a）(i)

From :

$$
z_1^{(i)} = W_1 x^{(i)} + b_1,\quad z_1^{(i)} \in \mathbb{R}^{D_{a_1}\times 1}
$$

we have :

- $W_1 \in \mathbb{R}^{D_{a_1}\times D_x}$
- $b_1 \in \mathbb{R}^{D_{a_1}\times 1}$

From :

$$
z_2^{(i)} = W_2 a_1^{(i)} + b_2,\quad z_2^{(i)} \in \mathbb{R}^{1\times 1}
$$

we have :

- $W_2 \in \mathbb{R}^{1\times D_{a_1}}$
- $b_2 \in \mathbb{R}^{1\times 1}$

(ii)

- $X \in \mathbb{R}^{D_x\times m}$ (each column is one example)
- $Y \in \mathbb{R}^{1\times m}$
- $Z_1 \in \mathbb{R}^{D_{a_1}\times m}$
- $A_1 \in \mathbb{R}^{D_{a_1}\times m}$
- $Z_2 \in \mathbb{R}^{1\times m}$
- $\hat Y \in \mathbb{R}^{1\times m}$

---

(b)

Since $\hat Y = \sigma(Z_2)$, with $\sigma'(z) = \sigma(z)(1-\sigma(z))$, we have :

$$
\frac{\partial J}{\partial Z_2}
= \frac{\partial J}{\partial \hat Y} \odot \frac{\partial \hat Y}{\partial Z_2}
= \frac{\partial J}{\partial \hat Y} \odot \hat Y \odot (1-\hat Y)
$$

where $\odot$ denotes elementwise product.

Plugging in the given expression:

$$
\frac{\partial J}{\partial Z_2}
= -\frac1m\left(
\frac{Y}{\hat Y} - \frac{1-Y}{1-\hat Y}
\right) \odot \hat Y \odot (1-\hat Y)
$$

Elementwise, for each entry,

$$
-\left(
\frac{Y}{\hat Y} - \frac{1-Y}{1-\hat Y}
\right)\hat Y(1-\hat Y)
= -\big(Y(1-\hat Y) - (1-Y)\hat Y\big)
= \hat Y - Y
$$

Therefore, in matrix form,

$$
\frac{\partial J}{\partial Z_2}
= \frac1m(\hat Y - Y)
$$

with shape $1\times m$.

Now, $Z_2 = W_2 A_1 + b_2$, so each column is $z_2^{(i)} = W_2 a_1^{(i)} + b_2.$ In matrix calculus, the gradient of $J$ with respect to $W_2$ is

$$
\frac{\partial J}{\partial W_2}
= \frac{\partial J}{\partial Z_2} A_1^\top
= \frac1m(\hat Y - Y) A_1^\top
$$

where

- $\frac{\partial J}{\partial Z_2} \in \mathbb{R}^{1\times m}$
- $A_1^\top \in \mathbb{R}^{m\times D_{a_1}}$

so $\frac{\partial J}{\partial W_2} \in \mathbb{R}^{1\times D_{a_1}}$ matching the shape of $W_2$.

Gradient descent update rule for $W_2$ with learning rate $\eta$:

$$
W_2 \leftarrow W_2 - \eta \frac{\partial J}{\partial W_2}
= W_2 - \eta \cdot \frac1m(\hat Y - Y) A_1^\top
$$

---

### Q2

(a)

Notice that

$$
\{\forall h:\,|\varepsilon(h)-\hat\varepsilon(h)|\le\gamma\}
$$

is the complementary set of:

$$
\{\exists h:\,|\varepsilon(h)-\hat\varepsilon(h)|>\gamma\}
$$

Thus,

$$
\Pr(\forall h:\,|\varepsilon(h)-\hat\varepsilon(h)|\le\gamma)
= 1 - \Pr(\exists h:\,|\varepsilon(h)-\hat\varepsilon(h)|>\gamma)
$$

Use the bound ($\star$)，If RHS $\ge 1-\delta$，then,

$$
\Pr(\exists h:\,|\varepsilon(h)-\hat\varepsilon(h)|>\gamma)
\le \delta
$$

Because of ($\star$)，a sufficient condition is:

$$
2k\exp(-2\gamma^2 n)\le \delta
$$

and we can get:

$$
\begin{aligned}
2k\exp(-2\gamma^2 n)\le \delta
&\iff \exp(-2\gamma^2 n)\le \frac{\delta}{2k}\\
&\iff -2\gamma^2 n \le \log\frac{\delta}{2k}\\
&\iff 2\gamma^2 n \ge \log\frac{2k}{\delta}\\
&\iff
n \ge \frac{1}{2\gamma^2}\log\frac{2k}{\delta}
\end{aligned}
$$

Thus, a required condition is:

$$
{
n \;\ge\; \frac{1}{2\gamma^2}\log\frac{2k}{\delta}
}
$$

(b)

Since (a), when:

$$
n \ge \frac{1}{2\gamma^2}\log\frac{2k}{\delta}
$$

the probability of 

$$
|\varepsilon(h)-\hat\varepsilon(h)|\le \gamma
$$

is at least $1-\delta$ for every  $h\in H$

for $h=\hat h$

$$
\varepsilon(\hat h)\le \hat\varepsilon(\hat h)+\gamma

$$

for $h=h^\ast$

$$
\hat\varepsilon(h^\ast)\le \varepsilon(h^\ast)+\gamma
$$

Since $\hat h$ is the ERM:

$$
\hat\varepsilon(\hat h)\le \hat\varepsilon(h^\ast)
$$

Then we can get:

$$
\varepsilon(\hat h)
\;\le\; \hat\varepsilon(\hat h)+\gamma
\;\le\; \hat\varepsilon(h^\ast)+\gamma
\;\le\; \varepsilon(h^\ast)+2\gamma
$$

Let

$$
\gamma = \sqrt{\frac{1}{2n}\log\frac{2k}{\delta}}
$$

Then

$$
\varepsilon(\hat h)
\;\le\; \varepsilon(h^\ast) + 2\sqrt{\frac{1}{2n}\log\frac{2k}{\delta}}
$$

holds with a at least $1-\delta$ probablity


