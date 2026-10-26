---
tags: [comps, numerical-analysis, FEM, MTH651]
---

# MTH 651 – Advanced Numerical Analysis: Comps Summary

Course theme: **finite element methods (FEM) for elliptic PDEs** (model problem: Poisson), the functional-analytic framework that makes them work, error theory, implementation (assembly, sparsity), and iterative solvers.

> Conventions: $(\cdot,\cdot)$ = $L^2$ inner product, $\|\cdot\|_0$ = $L^2$ norm, $\|\cdot\|_k$ / $|\cdot|_k$ = $H^k$ norm / seminorm, $h$ = max mesh size, $C$ = generic constant independent of $h$ and $u$.
> Items marked 📝 come from homework and add to (or sharpen) the lectures.

**Roadmap**

1. Warm-up: finite differences, rates of convergence
2. Variational formulation & Galerkin method (1D)
3. 1D error analysis (energy, $L^2$, max norm)
4. 2D Poisson, Sobolev/Hilbert spaces, trace theorem
5. Abstract elliptic theory (Riesz, Lax–Milgram-type result), Céa's lemma
6. Finite elements formally, assembly, affine maps, quadrature
7. Interpolation theory & error estimates, regularity
8. $L^2$ estimates (Aubin–Nitsche), Neumann BCs
9. Sparse matrices & direct solvers
10. Iterative solvers (stationary methods, Richardson, Jacobi, GS)
11. Homework highlights
12. Quick-reference cheat sheet

---

## 1. Warm-up: Finite Differences and Convergence Rates

### 1.1 Model problem: diffusion

Conservation law on $[a,b]$ plus Fourier's law $F=-c\,q_x$ gives the **heat equation**
$$q_t = c\,q_{xx} + f.$$

Steady state ($q_t=0$) gives the **Poisson problem**
$$-\Delta u = f \ \text{ in } \Omega, \qquad u=u_D \ \text{ on } \partial\Omega,
\qquad \Delta u=\sum_{i=1}^d \partial_{x_i}^2 u.$$

Poisson is linear and self-adjoint, and is the prototype for the rest of the course.

### 1.2 Two-step view of numerical PDE

1. **Discretization**: infinite-dimensional PDE $\to$ finite algebraic system. System size is tied to accuracy.
2. **Solution**: solve the (large, sparse) system, with direct or iterative methods.

### 1.3 Difference quotients (Taylor expansion)
$$D_Ff=\frac{f(x+h)-f(x)}{h},\quad D_Bf=\frac{f(x)-f(x-h)}{h},\quad D_Cf=\frac{f(x+h)-f(x-h)}{2h}.$$

| Quotient | Error |
|---|---|
| Forward / backward | $\mathcal O(h)$ |
| Centered | $\mathcal O(h^2)$ |

Second derivative: $D^2f(x)=\dfrac{f(x-h)-2f(x)+f(x+h)}{h^2}$, with error $\mathcal O(h^2)$.

### 1.4 Order of accuracy and empirical rates

If $e(h)\sim h^p$ the method is **$p$-th order**, and $e(rh)\sim r^p e(h)$. Halving $h$ divides the error by $2^p$.

**Empirical rate** from two runs $(h_1,e_1),(h_2,e_2)$:
$$p\approx \frac{\log(e_1/e_2)}{\log(h_1/h_2)}.$$

On a log–log plot of $e$ against $h$, the slope is $p$. At about $10^{-16}$ round-off dominates and the error stagnates.

### 1.5 1D finite differences for Poisson

$-u''=f$, $u(0)=u(1)=0$. Take grid $x_i=ih$, $h=1/N$, and solve $-D^2u_i=f(x_i)$ for $1\le i\le N-1$. This gives a tridiagonal matrix $\frac1{h^2}\,\mathrm{tridiag}(-1,2,-1)$.

- Consistency error is $\mathcal O(h^2)$, and the observed convergence is 2nd order.
- Consistency alone does not prove convergence. That needs stability (a proof still has to be done).
- FEM is preferred because it gives systematic analysis, unstructured meshes and complicated geometry, adaptivity, and it mimics PDE structure (conservation, etc.).

---

## 2. Variational Formulation & Galerkin Method (1D)

### 2.1 Three equivalent problems

Setting: $-u''=f$ on $[0,1]$, $u(0)=u(1)=0$.
$$V=\{v\text{ continuous, } v' \text{ bounded}, v(0)=v(1)=0\}.$$

- **(D)** Differential: $-u''=f$.
- **(V)** Variational (weak): find $u\in V$ with $a(u,v)=(f,v)$ for all $v\in V$, where $a(u,v)=(u',v')$.
- **(M)** Minimization: find $u\in V$ minimizing the **energy** $F(v)=\tfrac12 a(v,v)-(f,v)$.

(Note: $F$ is quadratic, not linear.)

**Key equivalences:**
- (V) $\Leftrightarrow$ (M). For (V)$\Rightarrow$(M), write $F(u+w)=F(u)+\tfrac12 a(w,w)\ge F(u)$.
- (D) $\Rightarrow$ (V). Multiply by a **test function** $v$, integrate, **integrate by parts**, and boundary terms vanish since $v(0)=v(1)=0$.
- (V) $\Rightarrow$ (D) if $u\in C^2$. Integrate by parts back to get $(u''+f,v)=0$ for all $v$, so $u''+f=0$.
- **Uniqueness of (V)**: if $u_1,u_2$ both solve it, take $v=u_1-u_2$. Then $\|(u_1-u_2)'\|_0=0$, so $u_1-u_2$ is constant and zero by the BC.

### 2.2 Galerkin method

Replace $V$ by a finite-dimensional subspace $S\subset V$ with basis $\{\phi_j\}_{j=1}^N$. Find $u_S\in S$ with $a(u_S,v_S)=(f,v_S)$ for all $v_S\in S$. Writing $u_S=\sum u_j\phi_j$:
$$A\mathbf u=\mathbf b,\qquad A_{ij}=a(\phi_i,\phi_j)\ (\text{stiffness matrix}),\qquad b_i=(f,\phi_i)\ (\text{load vector}).$$

$A$ is **symmetric positive definite (SPD)**, inherited from $a(\cdot,\cdot)$. FEM is the choice of $S$ with locally supported basis functions, in a systematic sequence $S_1,S_2,\dots$ with $u_{S_k}\to u$.

### 2.3 Piecewise linear space in 1D

Partition $0=x_0<\dots<x_{n+1}=1$. Take $S=\{v\in C^0:\ v|_{[x_i,x_{i+1}]}\text{ linear},\ v(0)=v(1)=0\}$, with $\dim S=n$.

- **Hat (nodal) basis**: $\phi_i(x_j)=\delta_{ij}$. The coefficients of $v$ are its **nodal values**.
- **Nodal interpolant**: $f_I=\sum f(x_i)\phi_i$.
- Support of $\phi_i$ is two intervals, so $A$ is **tridiagonal**.
- Meshes need not be uniform (an advantage over FD).

### 2.4 Galerkin orthogonality and best approximation
$$a(u-u_S,v_S)=0\quad\forall v_S\in S.$$

**Theorem (best approximation in energy norm).** With $\|v\|_A=a(v,v)^{1/2}$ (the energy norm; here $\|v'\|_0$):
$$\|u-u_S\|_A\le\|u-v_S\|_A\quad \forall v_S\in S.$$

So $u_S$ is the $a$-orthogonal projection of $u$ onto $S$.

**Cauchy–Schwarz**: $|(u,v)|\le\|u\|\,\|v\|$, with equality iff $u,v$ are collinear.

---

## 3. 1D Error Analysis

### 3.1 Energy norm

Interpolation estimate (1D, piecewise linear):
$$\|u-u_I\|_A\le C\,h\,\|u''\|_0.$$

Combined with best approximation:
$$\|u-u_S\|_A\le C h\|u''\|_0=Ch\|f\|_0.$$

First-order in the energy norm.

### 3.2 $L^2$ norm via duality (Aubin–Nitsche)

Dual problem: $-w''=u-u_h$, $w(0)=w(1)=0$. Then
$$\|u-u_h\|_0^2=a(u-u_h,w)=a(u-u_h,w-v)\le\|u-u_h\|_A\inf_v\|w-v\|_A.$$

Result:
$$\boxed{\|u-u_h\|_0\le Ch^2\|u''\|_0.}$$

Second-order in $L^2$.

### 3.3 Max norm and the Green's function (1D special)

Green's function:
$$g_s(x)=\begin{cases}x(1-s)&x<s\\ s(1-x)&x\ge s\end{cases}$$

It satisfies $a(v,g_s)=v(s)$ for all $v\in V$.

Since $g_{x_i}\in S$, Galerkin orthogonality gives $u(x_i)-u_S(x_i)=a(u-u_S,g_{x_i})=0$. So **in 1D the FE solution is nodally exact**: $u_S=u_I$. Then
$$\|u-u_S\|_{\max}\le Ch^2\|u''\|_{\max}.$$

---

## 4. 2D/3D Poisson, Sobolev Spaces, Traces

### 4.1 Variational form in $\Omega\subset\mathbb R^d$

Integration by parts (Green's identity):
$$\int_K(\nabla\cdot\mathbf u)\,v\,dx=-\int_K\mathbf u\cdot\nabla v\,dx+\int_{\partial K}(\mathbf u\cdot\mathbf n)v\,ds.$$

Special case $v\equiv1$ is the **divergence theorem**. Apply it to $-\Delta u=f$:
$$\text{find }u\in H^1_0(\Omega):\quad (\nabla u,\nabla v)=(f,v)\quad\forall v\in H^1_0(\Omega).$$

This is equivalent to minimizing $\tfrac12(\nabla v,\nabla v)-(f,v)$.

### 4.2 Function spaces

- **Banach space**: complete normed space (Cauchy sequences converge).
- **Hilbert space**: Banach space whose norm comes from an inner product.
- **$L^p(\Omega)$**: $\|f\|_{L^p}=(\int|f|^p)^{1/p}<\infty$. Example: $x^{-\beta}\in L^p(0,1)$ iff $\beta<1/p$. Elements are equivalence classes (a.e. equal), so point values are not defined.
- **Weak derivative** $D^\alpha_w u$: the function satisfying
$$\int_\Omega D^\alpha_w u\,\phi\,dx=(-1)^{|\alpha|}\int_\Omega u\,D^\alpha\phi\,dx\quad\forall\phi\in C_c^\infty(\Omega).$$
It is unique if it exists, and it agrees with the classical derivative when that exists.
- **Sobolev space** $W^{k,p}(\Omega)=\{u\in L^p: D^\alpha_wu\in L^p,\ |\alpha|\le k\}$ with norm $\big(\sum_{|\alpha|\le k}\|D_w^\alpha u\|_{L^p}^p\big)^{1/p}$.
- **$H^k=W^{k,2}$** is a Hilbert space with $(u,v)_k=\sum_{|\alpha|\le k}(D^\alpha u,D^\alpha v)_0$.
- **$H^1$ seminorm** $|u|_1=\|\nabla u\|_0$ is the energy norm.
- Example: $|x|\in W^{1,p}(-1,1)$ with weak derivative $\operatorname{sign}(x)$. A discontinuous function is not in $W^{1,p}$.

### 4.3 Trace theorem

$u|_{\partial\Omega}$ is not defined pointwise for $L^p$ functions, but the trace operator $\gamma:W^{1,p}(\Omega)\to L^p(\partial\Omega)$ exists, is bounded, and agrees with restriction for smooth $u$.
$$H^1_0(\Omega)=\{v\in H^1(\Omega):\gamma(v)=0\}.$$

### 4.4 Continuity $\Leftrightarrow$ $H^1$ for piecewise polynomials

**Proposition.** If $u$ is piecewise smooth on a mesh $\mathcal T$, then
$$u\in H^1(\Omega)\iff u\ \text{is globally continuous}.$$

(Proof idea: elementwise integration by parts leaves an interface term $\int_\Gamma[u]\cdot\boldsymbol\phi$, where $[u]=u_1\mathbf n_1+u_2\mathbf n_2$ is the jump across the mesh skeleton $\Gamma$. This vanishes for all $\phi$ iff $[u]=0$.)

Consequences:
- $W_h$ = piecewise linear, possibly discontinuous, so $W_h\subset L^2$ but $W_h\not\subset H^1$.
- $V_h$ = continuous piecewise linear, so $V_h\subset H^1$ and it is a valid conforming FE space.
- FE spaces are subspaces of $H^1$, so $C^0$ continuity is required.

---

## 5. Abstract Elliptic Theory

### 5.1 Riesz representation theorem

For a Hilbert space $H$ and a bounded linear functional $L$, there is a unique $u\in H$ with
$$L(v)=(u,v)\ \ \forall v,\qquad\|L\|=\|u\|.$$

Here $\|L\|=\sup_{v\ne0}|L(v)|/\|v\|$, and a linear functional is continuous iff bounded.

### 5.2 Abstract problem (Lax–Milgram for symmetric $a$)

Let $V$ be a Hilbert space and let $a:V\times V\to\mathbb R$ be bilinear with:
1. **Symmetric**.
2. **Continuous**: $|a(u,v)|\le\gamma\|u\|_V\|v\|_V$.
3. **Coercive ($V$-elliptic)**: $a(v,v)\ge\alpha\|v\|_V^2$.
4. $L$ linear and **continuous**: $|L(v)|\le\Lambda\|v\|_V$.

**Theorem.** The problem $a(u,v)=L(v)$ for all $v$ is well-posed:
- A unique solution exists.
- **Stability**: $\|u\|_V\le\Lambda/\alpha$.
- It is equivalent to minimizing $F(v)=\tfrac12a(v,v)-L(v)$.

(Proof idea: properties 1–3 make $a$ an inner product equivalent to $\|\cdot\|_V$. Apply Riesz in the $a$-inner product.)

### 5.3 Applying it to Poisson

$V=H^1_0$, $a=(\nabla u,\nabla v)$, $L=(f,v)$. Properties 1, 2, 4 follow from Cauchy–Schwarz. Coercivity needs the **Poincaré (Friedrichs) inequality**:
$$\|v\|_0^2\le C_\Omega\|\nabla v\|_0^2\quad\forall v\in H^1_0(\Omega).$$

Then $a(v,v)\ge\frac{1}{C_\Omega+1}\|v\|_1^2$. In 1D, $\alpha=1/2$. The Dirichlet BC is essential here.

### 5.4 Discrete problem and Céa's lemma

For $V_h\subset V$ with $\dim V_h=M$:
- $A$ is SPD (from symmetry and coercivity), so $A\mathbf u=\mathbf b$ has a unique solution.
- Discrete stability: $\|u_h\|_V\le\Lambda/\alpha$.
- Galerkin orthogonality: $a(u-u_h,v_h)=0$.

**Theorem (Céa's lemma / quasi-optimality).**
$$\boxed{\|u-u_h\|_V\le\frac{\gamma}{\alpha}\inf_{v_h\in V_h}\|u-v_h\|_V.}$$

The FE solution is the best approximation in the energy norm. Up to the constant $\gamma/\alpha$, it is the best in the $V$-norm. (Proof: $\alpha\|e\|^2\le a(e,e)=a(e,u-v_h)\le\gamma\|e\|\|u-v_h\|$.) The remaining task is to bound the approximation error $\inf_{v_h}\|u-v_h\|$, which is done by interpolation.

---

## 6. Finite Elements: Definition, Assembly, Implementation

### 6.1 Formal definition (Ciarlet triple)

A **finite element** is $(K,P_K,\Sigma_K)$:
- $K$: geometric cell (triangle, interval, quad, ...).
- $P_K$: finite-dimensional local function space (e.g. $P^1$ with $\dim=3$ on a triangle).
- $\Sigma_K=\{\sigma_i\}$: **degrees of freedom (DOFs)**, linear functionals on $P_K$.

**Unisolvence**: $\boldsymbol\sigma:P_K\to\mathbb R^n$ is a bijection, so DOF values determine the function uniquely.
- DOFs form a basis of $P_K^*$.
- The **dual (nodal) basis** $\{\phi_i\}$ satisfies $\sigma_j(\phi_i)=\delta_{ij}$. For vertex-value DOFs these are the hat functions (Lagrange basis).
- **Local interpolant** $\Pi_K u\in P_K$: $\sigma_i(\Pi_Ku)=\sigma_i(u)$.

Examples:
- Linear triangle: DOFs are vertex values.
- Quadratic triangle: DOFs are vertex and edge-midpoint values (6 DOFs).

### 6.2 Local-to-global

The space $V_h=\{v_h: v_h|_K\in P_K,\ \text{continuity via shared DOFs}\}$ has global DOFs $\Sigma=\bigcup_K\Sigma_K$, a global basis, and local basis functions given by restriction.

### 6.3 Assembly

- Support of $\phi_i$ = union of elements containing vertex $i$. So $A_{ij}\neq0$ only if $i,j$ share an element, and $A$ is **sparse**: nonzeros per row $=$ vertex **valence**, which is $\mathcal O(1)$. Hence $\mathrm{nnz}(A)=\mathcal O(n)$.
- Local stiffness: $(A_K)_{ij}=\int_K\nabla\phi_{I_i}\cdot\nabla\phi_{I_j}\,dx$ (a $3\times3$ matrix).
- **Assemble**: $A=\sum_K\hat A_K$, where each $A_K$ is scattered into the global matrix by vertex indices. The load vector $b$ is assembled the same way.

### 6.4 Affine equivalence / reference element

Every triangle is $K=T(\hat K)$ with $\hat K$ the reference triangle $(0,0),(1,0),(0,1)$ and
$$T(\hat{\mathbf x})=\mathbf a+J\hat{\mathbf x}\quad(\text{affine}).$$

Pull integrals back:
$$\int_K f\,dx=\int_{\hat K}f(T(\hat{\mathbf x}))\,|\det J|\,d\hat{\mathbf x},$$

and basis functions on $K$ are $\phi=\hat\phi\circ T^{-1}$, so gradients transform via $J^{-T}$.

### 6.5 📝 Explicit P1 triangle formulas (Homework 3, 4)

Let $K$ have vertices $(x_i,y_i)$, $i=1,2,3$.
$$|K|=\tfrac12\Big|\det\begin{pmatrix}x_1&y_1&1\\x_2&y_2&1\\x_3&y_3&1\end{pmatrix}\Big|.$$

Gradients of the barycentric (hat) functions are constant on $K$. With
$$G=\begin{pmatrix}1&1&1\\x_1&x_2&x_3\\y_1&y_2&y_3\end{pmatrix}^{-1}\begin{pmatrix}0&0\\1&0\\0&1\end{pmatrix}\quad(\text{rows } =\nabla\phi_i),$$
$$\boxed{A_K=|K|\,GG^T}\qquad(\text{local stiffness}).$$

**Local mass matrix** $M_{ij}=\int\phi_i\phi_j$:
$$M_K=\frac{|K|}{12}\begin{pmatrix}2&1&1\\1&2&1\\1&1&2\end{pmatrix}.$$

**One-point (centroid) quadrature**: $\int_Kf\approx|K|f(\xi)$ with $\xi=\tfrac13(x_1+x_2+x_3)$. Since $\phi_i(\xi)=\tfrac13$, $\int_Kf\phi_i\approx\tfrac13|K|f(\xi)$.

**Dirichlet BC elimination**: set the boundary rows and columns of $A$ to identity and the boundary entries of $F$ to 0 (for homogeneous data).

**Mesh data structure**: vertices $V\in\mathbb R^{N\times2}$, triangles $T\in\mathbb Z^{n_t\times3}$ (indices into $V$).

### 6.6 Numerical quadrature

FE integrals (e.g. $\int f\phi_i$) are approximated by $\int_{-1}^1F\approx\sum_iw_iF(x_i)$. Map to $[a,b]$ via $T(x)=\tfrac12(b-a)(x+1)+a$:
$$\int_a^bF\,dx\approx\tfrac12(b-a)\sum_iw_iF(T(x_i)).$$

- If $Q$ is exact for polynomials of degree $\le p$, the error on an interval of length $h$ is $\mathcal O(h^{p+2})$. On a subdivided fixed interval the **global** error is $\mathcal O(h^{p+1})$.
- $n$ arbitrary points can be exact up to degree $n-1$.
- **Gauss–Legendre** with $n$ points is exact to degree $2n-1$, with interior points. Two points: $\pm1/\sqrt3$, weights $1,1$.
- Gauss–Radau (one endpoint) is exact to degree $2n-2$. Gauss–Lobatto (both endpoints) is exact to degree $2n-3$.
- Related to orthogonal polynomials.

---

## 7. Interpolation Theory and Error Estimates

Reduce to bounding $\|u-\Pi_hu\|$, since Céa gives $\|u-u_h\|_V\le\frac\gamma\alpha\inf_{v_h}\|u-v_h\|_V$.

### 7.1 Mesh geometry

- $h_K=\mathrm{diam}(K)$.
- $\rho_K$ = diameter of the inscribed circle.
- $h=\max_Kh_K$.
- **Shape-regular** family: $\rho_K/h_K\ge\beta>0$ uniformly. This is a minimum-angle bound.

### 7.2 Linear interpolation on a triangle

**Theorem.** For $v\in C^2(K)$ and $v_h=\Pi_Kv\in P^1$:
$$\|v-v_h\|_{L^\infty(K)}\le 2h_K^2\max_{|\alpha|=2}\|D^\alpha v\|_{L^\infty(K)},$$
$$\max_{|\alpha|=1}\|D^\alpha(v-v_h)\|_{L^\infty(K)}\le 6\frac{h_K^2}{\rho_K}\max_{|\alpha|=2}\|D^\alpha v\|_{L^\infty(K)}.$$

Proof ingredients (not detailed here):
- Taylor expand at each vertex.
- The barycentric identities $\sum\lambda_i=1$, $\sum\nabla\lambda_i=0$, and $\sum_i p_i\lambda_i=0$.
- The bound $|\partial\lambda_i/\partial x_j|\le1/\rho_K$.

**Sobolev version** ($v\in H^2(K)$):
$$\|v-v_h\|_{0,K}\le Ch_K^2|v|_{2,K},\qquad |v-v_h|_{1,K}\le C\frac{h_K^2}{\rho_K}|v|_{2,K}.$$

Summing over elements with shape regularity:
$$\|v-\Pi_hv\|_0\lesssim h^2|v|_2,\qquad |v-\Pi_hv|_1\lesssim h|v|_2.$$

### 7.3 Higher degree $r$ and lower regularity

For degree-$r$ piecewise polynomials and $v\in H^{r+1}$:
$$\|v-\Pi_hv\|_0\lesssim h^{r+1}|v|_{r+1},\qquad |v-\Pi_hv|_1\lesssim h^r|v|_{r+1}.$$

If $v\in H^s$ with $1<s\le r+1$:
$$\|v-\Pi_hv\|_0\lesssim h^s|v|_s,\qquad|v-\Pi_hv|_1\lesssim h^{s-1}|v|_s.$$

So the rate is $\min(r,s-1)$ in $H^1$, and low regularity caps the rate.

### 7.4 Regularity of the exact solution

For smooth $\partial\Omega$ (elliptic regularity):
$$\|u\|_{s+2}\le C\|f\|_s.$$

The solution gains two derivatives over $f$. This can fail on non-smooth boundaries (e.g. re-entrant corners).

### 7.5 Final a priori estimates

With $V_h$ of degree $r=p$ and $u\in H^{p+1}$:
$$\boxed{\|u-u_h\|_{H^1}\le Ch^p\|u\|_{H^{p+1}}}$$

(optimal order). Linear elements ($p=1$) give $\mathcal O(h)$ in $H^1$.

---

## 8. $L^2$ Estimates and Neumann Boundary Conditions

### 8.1 Duality (Aubin–Nitsche) argument

Let $e=u-u_h$ and solve the auxiliary problem
$$-\Delta w=e\ \text{in }\Omega,\qquad w=0\ \text{on }\partial\Omega,\qquad\|w\|_{H^2}\le C\|e\|_{L^2}\ (\text{elliptic regularity}).$$

Then
$$\|e\|_0^2=a(e,w)=a(e,w-\Pi_hw)\le C\|e\|_1\,h\|w\|_2\le Ch\|e\|_1\|e\|_0.$$

So $\|e\|_0\le Ch\|e\|_1$, and with the $H^1$ estimate
$$\boxed{\|u-u_h\|_{L^2}\le Ch^{p+1}\|u\|_{H^{p+1}}.}$$

The $L^2$ rate is one order higher than the $H^1$ rate. The argument needs $H^2$-regularity of the dual problem (convex or smooth domain).

| Elements | $H^1$ / energy | $L^2$ |
|---|---|---|
| $P^1$ | $h$ | $h^2$ |
| $P^2$ | $h^2$ | $h^3$ |
| $P^p$ | $h^p$ | $h^{p+1}$ |

### 8.2 Neumann boundary conditions

Neumann (flux) condition: $\partial_nu=\nabla u\cdot\mathbf n=0$ on $\partial\Omega$ (insulated boundary).

**Problem with reaction term:** $-\Delta u+u=f$, $\partial_nu=0$.

Multiply by $v$ and integrate by parts. The boundary term $\int_{\partial\Omega}(\partial_nu)v$ vanishes, giving
$$\text{find }u\in H^1(\Omega):\ (\nabla u,\nabla v)+(u,v)=(f,v)\quad\forall v\in H^1(\Omega).$$

- Here $a(\cdot,\cdot)=(\cdot,\cdot)_1$ exactly, so symmetry, continuity and coercivity hold with $\gamma=\alpha=1$.
- The space is $H^1$, not $H^1_0$.
- Neumann BCs are **natural** BCs: they are implicit in the weak form. Dirichlet BCs are **essential** BCs: built into the space.

**Pure Neumann problem $-\Delta u=f$, $\partial_nu=0$** is not well-posed on $H^1$:
- $u$ is unique only up to an additive constant.
- A solution exists only if the compatibility condition $\int_\Omega f=0$ holds, since $\int f=-\int\Delta u=\int\nabla u\cdot\nabla1=0$.

Fix: use the mean-zero space $V=\{v\in H^1:\int_\Omega v=0\}$. The **mean Poincaré inequality** $\|u\|_0\le C\|\nabla u\|_0$ for $\bar u=0$ gives coercivity. The weak problem is then well-posed. For $f$ without mean zero, $(f,v)=(f-\bar f,v)$ on $V$, so the solution solves the PDE with $\tilde f=f-\bar f$.

---

## 9. Sparse Matrices and Direct Solvers

### 9.1 Complexity of dense operations ($n\times n$)

| Operation | Flops | Memory |
|---|---|---|
| Mat–vec $A\mathbf v$ | $2n^2$ ($\mathcal O(n^2)$) | $\mathcal O(n^2)$ |
| Mat–mat $AB$ (triple loop) | $2n^3$ ($\mathcal O(n^3)$) | $3n^2$ |

Unusable for $n\sim10^6$–$10^9$. FE matrices are sparse: $\mathrm{nnz}(A)=\mathcal O(n)$.

### 9.2 Sparse formats

- **COO**: arrays $(I,J,V)$ listing each nonzero; memory $3\,\mathrm{nnz}$. Mat–vec cost $\mathcal O(\mathrm{nnz})$.
- **CSR** (compressed sparse row): $I$ is a **row pointer** of length $m+1$ and row $i$'s entries occupy indices $[I[i],I[i+1])$ of $J,V$.
  - Nonzeros in row $i$ is $I[i+1]-I[i]$, and $I[m]=\mathrm{nnz}$.
  - Memory $2\,\mathrm{nnz}+m+1$.
  - Fast row access and mat–vec: for each row, $w_i=\sum_{k=I[i]}^{I[i+1]-1}V[k]\,v_{J[k]}$, costing $\mathcal O(\mathrm{nnz})$.
- CSC is the column analogue.
- 0-based indexing is used in the notes.

**Sparse mat–mat $C=AB$ (CSR)**: $C_{ij}$ is potentially nonzero whenever $A_{ik}\ne0$ and $B_{kj}\ne0$ for some $k$. A two-pass algorithm is used:
1. Symbolic pass: compute $I_C$ and $\mathrm{nnz}(C)$, using a marker array to detect the first visit of each column $j$ in a row.
2. Numeric pass: fill $J_C,V_C$, accumulating repeated $(i,j)$.

### 9.3 LU / Cholesky and fill-in (Factorization notebook)

- LU by repeated **Schur complements**: with $S=\begin{pmatrix}\alpha&\mathbf a^T\\ \mathbf b&C\end{pmatrix}$, set $l=\mathbf b/\alpha$, $u^T=\mathbf a^T$, and recurse on $C-\mathbf b\mathbf a^T/\alpha$.
- For SPD matrices use **Cholesky** $A=LL^T$ (about half the cost, no pivoting needed).
- **Fill-in**: factors of a sparse matrix can be much denser than $A$. For banded matrices, fill stays within the band. Dense direct solves cost $\mathcal O(n^3)$, which motivates iterative methods.

---

## 10. Iterative Solvers

Setting: $A$ SPD (from a symmetric coercive form).

### 10.1 SPD facts

- A symmetric matrix is positive definite iff $v^TAv>0$ for $v\ne0$, iff all eigenvalues are positive. Eigenvalues of symmetric matrices are real, and $A=Q^T\Lambda Q$ by the spectral theorem.
- Condition number $\kappa(A)=\lambda_{\max}/\lambda_{\min}$.

### 10.2 Stationary methods
$$x^{(i+1)}=x^{(i)}+B\,(b-Ax^{(i)}),\qquad B\ \text{nonsingular, fixed}.$$

- Fixed point $\Leftrightarrow$ $Ax=b$. $B\approx A^{-1}$, and $B=A^{-1}$ gives convergence in one step.
- Error propagation: $e^{(i+1)}=(I-BA)e^{(i)}$. Convergent iff $\rho(I-BA)<1$, and $\rho$ is the asymptotic rate.
- Cost per iteration: one mat–vec with $A$ ($\mathcal O(\mathrm{nnz})$), one application of $B$, and vector updates.
- Residual $r=b-Ax$ and error $e=A^{-1}r$.

**Examples** (write $A=D+L_s+U_s$ for diagonal, strict lower and strict upper parts):

| Method | $B$ |
|---|---|
| Richardson | $B=\omega I$ |
| Jacobi | $B=D^{-1}$ |
| Gauss–Seidel | $B=(D+L_s)^{-1}$ (i.e. $L^{-1}$, $L$ = lower triangle including the diagonal) |

Gauss–Seidel: $Lx^{(i+1)}=b-Ux^{(i)}$ (use new components as soon as they are available).

### 10.3 Convergence theory

**Symmetric $B$**: $BA$ is self-adjoint in the $A$-inner product, so its eigenvalues are real, and
$$\rho(I-BA)=\max\{|1-\lambda_{\min}(BA)|,\ |1-\lambda_{\max}(BA)|\}.$$

The method converges iff $0<\lambda_{\min}(BA)\le\lambda_{\max}(BA)<2$.

**Symmetrized method** (for general $B$): half-step with $B$, then half-step with $B^T$. Then
$$I-\bar BA=(I-B^TA)(I-BA)=(I-BA)^*(I-BA),\qquad\bar B=B+B^T-B^TAB.$$

This is positive semidefinite in the $A$-inner product, so $\rho(I-\bar BA)=1-\lambda_{\min}(\bar BA)$.

**Theorem.** The symmetrized method converges iff $B^{-1}+B^{-T}-A$ is SPD. For general $B$, $\rho(I-BA)^2\le\|I-BA\|_A^2=\rho(I-\bar BA)$, so $\Phi_B$ converges iff $\Phi_{\bar B}$ does.

**Theorem (Richardson).** Converges iff $0<\omega<2/\lambda_{\max}(A)$. The optimal choice is
$$\omega_{\rm opt}=\frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad\text{rate}=\frac{\kappa(A)-1}{\kappa(A)+1}.$$

**Theorem (Jacobi).** Converges if $A$ is strictly diagonally dominant.

**Theorem (Gauss–Seidel).** Converges for every SPD $A$. (Reason: $B^{-1}+B^{-T}-A=D$ is SPD.)

### 10.4 Behavior on FE matrices (📝 Homework 4, plus lecture MFEM demos)

For a quasi-uniform 2D mesh with size $h$:

| Matrix | Spectrum | $\kappa$ | Consequence |
|---|---|---|---|
| Stiffness $A$ | $\lambda_{\max}=\mathcal O(1)$, $\lambda_{\min}=\mathcal O(h^2)$ | $\mathcal O(h^{-2})$ | Richardson / Gauss–Seidel iterations grow like $h^{-2}$ |
| Mass $M$ | $\lambda\in[ch^2,Ch^2]$ ($d=2$) | $\mathcal O(1)$ | iteration count independent of $h$ |

- Rate $\frac{\kappa-1}{\kappa+1}\approx1-\frac2\kappa$, so the number of iterations to reduce error by a fixed factor is $\mathcal O(\kappa)$.
- Richardson needs $\omega$ from the analytic eigenvalue estimates in terms of $h$ (do not compute eigenvalues).
- **Steepest descent** uses the same rate bound $\frac{\kappa-1}{\kappa+1}$ (in the $A$-norm) but picks the step $\omega_i=\frac{r^Tr}{r^TAr}$ adaptively, so no eigenvalue knowledge is needed.
- Gauss–Seidel iteration count on $A$ grows like $h^{-2}$ (about 4× per refinement in 2D). On $M$ it is about constant.
- Mass matrix $M_{ij}=\int\phi_i\phi_j$ is SPD and also sparse. Its sparsity pattern is the same as the stiffness pattern for P1. $M$ is the matrix for **$L^2$ projection**: $M\mathbf u=\mathbf b$ with $b_i=(f,\phi_i)$ gives the $L^2$-best approximation $u_h\in V_h$ of $f$.
- Timing: sparse iterative solves scale roughly linearly in $n$ per iteration, versus about $n^3$ for dense Gaussian elimination.
- Iterative methods that scale better (CG, preconditioning) are only mentioned: "(better:) (preconditioned) conjugate gradient".

---

## 11. Homework Highlights (items beyond the lectures)

### HW1 – 1D FEM solver
- Algorithm: mesh, load vector via two-point Gauss quadrature on each of the two intervals touching $\phi_i$, stiffness assembly, boundary rows set to identity.
- $a(\phi_i,\phi_j)\ne0$ only for $j\in\{i-1,i,i+1\}$.
- **Convergence study** (10 to 320 elements): expect energy rate 1 and $L^2$ rate 2.
- 📝 **One-point (midpoint) quadrature**: $\int_a^bf\approx(b-a)f(\tfrac{a+b}2)$.
  - Local error is $\mathcal O(h^3)$ (Taylor at the midpoint; the linear term integrates to zero). Global is $\mathcal O(h^2)$.
  - It is exact for degree $\le1$, so **stiffness entries $K_{ij}$ are computed exactly** (integrands $\phi_i'\phi_j'$ are constant).
- 📝 **Link to FD**: for a uniform mesh, the FE stiffness matrix is $\frac1h\,\mathrm{tridiag}(-1,2,-1)$, which is $h$ times the FD matrix $\frac1{h^2}\mathrm{tridiag}(-1,2,-1)$. Dividing by $h$, the FE system is the FD system with load $F_i/h=(f,\phi_i)/h\approx f(x_i)$. So P1 FEM on a uniform mesh reproduces the second-order FD scheme.
- 📝 **Non-uniform meshes**: FEM needs no changes beyond using local $h_i$ in the stiffness entries ($\frac1{h_i}+\frac1{h_{i+1}}$ on the diagonal, $-\frac1{h_i}$ off-diagonal). The FD stencil would have to be modified. Convergence rates remain $h$ (energy) and $h^2$ ($L^2$), with $h=\max h_i$.

### HW3 – 2D FEM solver
- Implement area, one-point quadrature, load vector and stiffness assembly via $A_K=|K|GG^T$, and Dirichlet elimination.
- Problem: $f\equiv1$ on a given mesh with boundary vertices detected via edge counting (boundary edges belong to exactly one triangle).
- **Convergence study** on the unit square with exact solution $u=\sin\pi x\sin\pi y$, so $f=2\pi^2\sin\pi x\sin\pi y$. Expect $L^2$ rate 2 for P1, energy rate 1. Note that one-point quadrature on $f$ is second-order accurate, so it does not degrade the $L^2$ rate.
- Textbook problems 4.1, 4.2, 4.8 (not reproduced).

### HW4 – Mass matrix and iterative solvers
- $M_K=\frac{|K|}{12}\begin{pmatrix}2&1&1\\1&2&1\\1&1&2\end{pmatrix}$, no BCs.
- Implement a generic stationary solver with relative residual tolerance $\|r\|_2/\|b\|_2\le10^{-5}$ (max 2000 iterations). Compare Richardson with optimal $\omega$, steepest descent, and Gauss–Seidel on $A$ and $M$ for $n_x=4,8,16,32$.
- Expected: iterations scale like $\kappa(A)\sim h^{-2}$ for the stiffness matrix and are $\mathcal O(1)$ for the mass matrix.
- Textbook problems 6.1, 7.5 (not reproduced).

---

## 12. Quick-Reference Cheat Sheet

### Key formulas

| Topic | Result |
|---|---|
| Variational problem | $a(u,v)=L(v)$ $\forall v\in V$ |
| Galerkin orthogonality | $a(u-u_h,v_h)=0$ |
| Stiffness / load | $A_{ij}=a(\phi_i,\phi_j)$, $b_i=L(\phi_i)$ |
| Well-posedness | symmetric, continuous ($\gamma$), coercive ($\alpha$), $L$ bounded ($\Lambda$); $\|u\|_V\le\Lambda/\alpha$ |
| Céa | $\|u-u_h\|_V\le\frac\gamma\alpha\inf_{v_h}\|u-v_h\|_V$ |
| Interpolation ($P^r$) | $\|v-\Pi v\|_0\lesssim h^{r+1}\|v\|_{r+1}$, $\|v-\Pi v\|_1\lesssim h^r\|v\|_{r+1}$ |
| A priori $H^1$ | $\|u-u_h\|_1\le Ch^p\|u\|_{p+1}$ |
| A priori $L^2$ (duality) | $\|u-u_h\|_0\le Ch^{p+1}\|u\|_{p+1}$ |
| Regularity | $\|u\|_{s+2}\le C\|f\|_s$ |
| Poincaré | $\|v\|_0\le C\|\nabla v\|_0$ on $H^1_0$ (or mean-zero $H^1$) |
| Trace | $H^1\to L^2(\partial\Omega)$; $H^1_0=\ker\gamma$ |
| Piecewise poly $\in H^1$ | iff globally continuous |
| Local stiffness P1 | $A_K=|K|GG^T$ |
| Local mass P1 | $M_K=\frac{|K|}{12}(\mathbf 1\mathbf 1^T+I)$ |
| Stationary iteration | $x^{k+1}=x^k+B(b-Ax^k)$, converges iff $\rho(I-BA)<1$ |
| Richardson | $\omega_{\rm opt}=\frac2{\lambda_{\min}+\lambda_{\max}}$, rate $\frac{\kappa-1}{\kappa+1}$ |
| GS | converges for all SPD $A$ |
| Jacobi | converges for strictly diagonally dominant $A$ |
| Symmetrized method | converges iff $B^{-1}+B^{-T}-A$ is SPD |
| CSR memory | $2\,\mathrm{nnz}+m+1$ |

### Concept checklist (likely exam questions)

1. Derive the weak form of Poisson with Dirichlet BCs. Show (D) $\Rightarrow$ (V) $\Leftrightarrow$ (M), and prove uniqueness.
2. Define Galerkin orthogonality. Prove best approximation in the energy norm and Céa's lemma.
3. State the hypotheses and conclusion of the abstract well-posedness theorem. Where does Poincaré enter?
4. Define weak derivative, $W^{k,p}$, $H^k$, $H^1_0$ (trace), and show that a piecewise polynomial is in $H^1$ iff continuous.
5. Define a finite element $(K,P_K,\Sigma)$ and unisolvence. Describe assembly, and the role of the reference element and affine map.
6. State the P1 interpolation estimates, the role of shape regularity ($\rho_K/h_K\ge\beta$), and the higher-order and low-regularity versions.
7. Do the duality argument for $L^2$. Where is $H^2$ regularity used?
8. Compare natural (Neumann) and essential (Dirichlet) BCs. Explain why pure Neumann fails and how to fix it.
9. Explain CSR and COO. Give mat–vec and storage costs, and why sparsity matters.
10. Define stationary methods and give convergence criteria (spectral radius, symmetrized criterion). Give the Richardson optimal $\omega$ and rate, and cite the Jacobi and Gauss–Seidel theorems.
11. Explain why the iteration count grows as $h^{-2}$ for $A$ but not for $M$ (condition numbers).
12. Gauss quadrature: degree of exactness $2n-1$, and the effect of quadrature error on FEM accuracy.
