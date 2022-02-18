
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
\mathbf{B}_{t+1} = \mathbf{B}_t + \begin{bmatrix} -\gamma_b p_{b, t} & -\gamma_g p_{g, t} \\ 1 & 0 \\ 0 & 1  \end{bmatrix} \mathbf{p}_t
$$
