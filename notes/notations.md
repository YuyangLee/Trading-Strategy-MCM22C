
# Notations

- Time step $t \in \{ 1, 2, \dots, 1826 \}$

- **Portfolio balance** of <ins>c</ins>ash, <ins>B</ins>TC and <ins>g</ins>old: **when day $t$ ends**: $\mathbf{B}_t = \begin{bmatrix} c_t & b_t & g_t \end{bmatrix}^T$
  - initially $\mathbf{B}_0 = \begin{bmatrix} c_0 & b_0 & g_0 \end{bmatrix}^T = \begin{bmatrix} 1000.0 & 0.0 & 0.0 \end{bmatrix}^T$

- **Trade price** of <ins>B</ins>TC and <ins>g</ins>old: $\mathbf{p}_t = \begin{bmatrix} p_{b, t} & g_{b, t} \end{bmatrix}^T$

- **Trade (in) amount** of <ins>B</ins>TC and <ins>g</ins>old: $\mathbf{s}_t = \begin{bmatrix} s_{b, t} & s_{g, t} \end{bmatrix}^T$

- **Discount of transection**: $\gamma = \begin{bmatrix} \gamma_b & \gamma_g \end{bmatrix}^T = 1 - \begin{bmatrix} \alpha_b & \alpha_g \end{bmatrix}^T$
  - as assumed $\gamma = \begin{bmatrix} \gamma_b & \gamma_g \end{bmatrix}^T = \begin{bmatrix} 0.98 & 0.99 \end{bmatrix}^T$ 

- **Tradability indicator** for gold: $\tau_t = \left\{ \begin{matrix} 1 & \mathrm{day\ } t \mathrm{\ is\ tradable} \\ 0 & \mathrm{else} \end{matrix} \right.$ 

Then we have

$$
\mathbf{B}_{t+1} = \mathbf{B}_t + \begin{bmatrix} -\gamma_b p_{b, t} & -\gamma_g p_{g, t} \\ 1 & 0 \\ 0 & 1  \end{bmatrix} \mathbf{s}_t
$$

# Optimizing Strategy with DRL 

We find the optimized strategy using Deep Reinforcement Learning.

## Formulation

### Environment

Consider $n_a$ types of assets (in the problem $n_a = 2$ for Gold BTC). Their prices at day $t$ are $\mathbf{p}_t \in \R_+^{n_a}$

The state consists of the current balances in the portfolio, as well as the a sequence of prices of the assets in the future:

$$
s_t = \begin{bmatrix} c_t & \mathbf{a}_t & \phi(\mathbf{p}_{t:t+l}) \end{bmatrix} \in \R_+^{2l + n_a + 1}
$$

where $\phi(\cdot)$ flattens its input into a $1-$ dimentional vector.

The sequence is predicted with a prices sequence in the past.

The actor determines the action $a_t = \begin{bmatrix} \Delta \mathbf{a}_t \end{bmatrix} \in \R^{2l+3}$ as the trading amount (positive for buying, negative for selling) of the assets, therefore

$$
s_{t+1} = s_t + \begin{bmatrix} - \gamma^{-1} \cdot \Delta \mathbf{a}_t & \Delta \mathbf{a}_t \end{bmatrix}
$$

The return of action $a_t$ is computed as

$$
r_t = W(s_{t+1}) - W(s_t)
$$

where the wealth function is:

$$
W(s_t) = c_t + \mathbf{p}_t \cdot \mathbf{a}_t
$$

The constraint for action $a_t$ is

$$
s_{t+1} \geq \mathbf{0}
$$

# Theories

Consider we have prices $p_{b, t}, p_{g, t}$ for BTC and Gold on day $t$, then $\eta_{b, t} = \frac{p_{b, t+1} - p_{b, t}}{p_{b, t}}, \eta_{g, t} = \frac{p_{g, t+1} - p_{g, t}}{p_{g, t}}$.

If we possses $b_t$ BTC and $g_t$ ounces of gold. We buy $\Delta b_t$ (sell $-\Delta b_t$) BTC and use all of them to buy gold. We'll get profit iff

$$
(b_t + \Delta b_t) p_{b, t} \eta_{b, t} + (g_t - \frac{\Delta b_t p_{b, t}}{p_{g, t}} \gamma_b \gamma_g) p_{g, t} \eta_{g, t} \geq b_t p_{b, t} \eta_{g, t} + g_t p_{g, t} \eta_{g, t}
$$

i.e.

$$
\Delta b_t p_{b, t} \eta_{b, t} + \frac{\Delta b_t p_{b, t}}{p_{g, t}} \gamma_b \gamma_g p_{g, t} \eta_{g, t} \geq 0
$$

When $p_{b, t} \eta_{b, t} +  p_{g, t} \eta_{g, t} \gamma_b \gamma_g > 0$:

$$
\Delta b_t \geq \frac{1}{p_{b, t} \eta_{b, t} +  p_{g, t} \eta_{g, t} \gamma_b \gamma_g}
$$

And when $p_{b, t} \eta_{b, t} +  p_{g, t} \eta_{g, t} \gamma_b \gamma_g < 0$:


$$
-\Delta b_t \geq \frac{1}{p_{b, t} \eta_{b, t} +  p_{g, t} \eta_{g, t} \gamma_b \gamma_g}
$$
