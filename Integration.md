**1) Observational Data and Trajectory Network**  
Given sampling timestamps $\{t_k\}_{k=1}^T$ and observations $x(t_k)\in\mathbb{R}^{N\times 3}$, a neural network is used to fit:
$$
\hat{x}(t;\theta)\in\mathbb{R}^{N\times 3}
$$

**2) Dynamics and Hyperedge Coupling**  
Base Rossler dynamics:
$$
f(x_i)=\begin{bmatrix}
-(y_i+z_i)\\
x_i + a_r y_i\\
b_r + z_i(x_i-c_r)
\end{bmatrix}
$$

Hyperedge coupling term (constructed based on the candidate hyperedge set):
$$
\Phi(x)\in\mathbb{R}^{N\times 3\times M},\quad A\in\mathbb{R}^{M\times 1}
$$

Total dynamics:
$$
\dot{x}(t)=f(x(t))+\Phi(x(t))A
$$

**3) Integral Constraint (Physics Loss)**  
For an arbitrary window $[t_1,t_2]$:
$$
\hat{x}(t_2;\theta)-\hat{x}(t_1;\theta)
\approx \int_{t_1}^{t_2}\Big(f(\hat{x}(t;\theta))+\Phi(\hat{x}(t;\theta))A\Big)\,dt
$$

Data fitting loss:
$$
\int_{t_1}^{t_2} g(t)\,dt \approx \frac{t_2-t_1}{2}\sum_{q=1}^Q w_q\, g\!\Big(\tfrac{t_1+t_2}{2}+\tfrac{t_2-t_1}{2}\xi_q\Big)
$$

**4) Loss Function**  

$$
\mathcal{L}_{data}=\frac1T\sum_{k=1}^T\|\hat{x}(t_k;\theta)-x(t_k)\|_2^2
$$

Physics integral loss (averaged over random windows):
$$
\mathcal{L}_{phys}=\mathbb{E}_{[t_1,t_2]}\Big\|
\hat{x}(t_2;\theta)-\hat{x}(t_1;\theta)
-\int_{t_1}^{t_2}\!(f(\hat{x})+\Phi(\hat{x})A)\,dt
\Big\|_2^2
$$

Sparsity regularization (Optional):
$$
\mathcal{L}_{sparse}=\|A\|_1
$$

Total objective:
$$
\min_{\theta, A}\ \mathcal{L}_{data}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{sparse}\mathcal{L}_{sparse}
$$

---

## Proof: Integral Constraints Are More Robust Under Smooth, Short-Correlated Errors

### Setting and Assumptions
Assume the true trajectory $x(t)$ satisfies
$$
\dot{x}(t)=f(x(t))+\Phi(x(t))A^\star.
$$
The network prediction obeys
$$
\hat{x}(t)=x(t)+\varepsilon(t),\quad \mathbb{E}[\varepsilon(t)]=0,
$$
and the error is **smooth and short-correlated** (correlation length $\ell_c$ is bounded), with no high-frequency oscillation. The candidate hyperedge dictionary is weakly identifiable: the design matrix has a large but finite condition number.

### Proposition (Robustness)
Under the assumptions above, the estimator $\hat{A}_I$ obtained from time-window integral constraints is less noise-sensitive than the pointwise (differential) estimator $\hat{A}_D$. Specifically, there exist constants $C>0$ and window length $T=t_2-t_1$ such that
$$
\mathbb{E}\|\hat{A}_I-A^\star\|^2 \le \frac{C}{T}\,\mathbb{E}\|\hat{A}_D-A^\star\|^2.
$$

### Proof Sketch

**Lemma 1 (Variance Reduction by Integration)**
Let $g(t)=g(x(t))$ be differentiable and define
$$
\Delta_I=\int_{t_1}^{t_2}\big[g(\hat{x}(t))-g(x(t))\big]dt.
$$
First-order expansion gives
$$
g(\hat{x}(t))\approx g(x(t))+J_g(x(t))\varepsilon(t),
$$
so
$$
\Delta_I\approx\int_{t_1}^{t_2}J_g(x(t))\varepsilon(t)\,dt.
$$
Under short-correlation,
$$
\mathrm{Var}(\Delta_I)=O(T\ell_c),
$$
which corresponds to an effective per-unit-time noise scale $O(\ell_c/T)$ that decreases with $T$.

**Lemma 2 (No Variance Reduction Pointwise)**
The pointwise residual
$$
\Delta_D=g(\hat{x}(t))-g(x(t))\approx J_g(x(t))\varepsilon(t)
$$
has $O(1)$ variance and does not decrease with $T$.

**Lemma 3 (Error Propagation Under Weak Identifiability)**
Linearize the estimator as
$$
Y=\Phi A^\star+\eta,\quad \hat{A}-A^\star\approx(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top\eta.
$$
With finite (though large) condition number, the second moment of $\hat{A}-A^\star$ scales with the noise variance.

**Combine the Lemmas**
The integral method replaces $\eta$ with a time-averaged $\tilde{\eta}$ whose variance shrinks as $O(\ell_c/T)$, while pointwise variance does not. Substituting into Lemma 3 yields
$$
\mathbb{E}\|\hat{A}_I-A^\star\|^2 \le \frac{C}{T}\,\mathbb{E}\|\hat{A}_D-A^\star\|^2.
$$
This proves the robustness advantage.

### Discussion
The result relies on smooth, short-correlated errors and a finite condition number. If errors contain strong high-frequency components or long-range correlation, or if the dictionary is severely unidentifiable (condition number diverges), the advantage may diminish or vanish.