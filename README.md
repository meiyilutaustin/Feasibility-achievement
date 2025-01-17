# Feasibility-achievement
## Introduction

Recent shifts toward machine learning (ML) aim to enhance the optimization efficiency of traditional iterative solvers. A notable challenge in applying ML methods to real-world problems is ensuring *feasibility*, that is, adhering to the physical and engineering constraints. 

Extensive research has been devoted to developing ML optimization proxies that ensure the feasibility of decisions. A commonly employed method is the incorporation of penalty terms aimed at minimizing constraint violations. Although this approach is applicable across various optimization problems, it merely penalizes rather than eliminates violations (soft rather than hard). Alternatively, some methods introduce a repair module that utilizes iterative algorithms to adjust solutions. These methods range from using commercial solvers to identify the closest feasible solution, to optimizing functions that minimize violations. Further advancements involve differentiable mapping functions that directly enforce constraint satisfaction.

This notebook visualizes the mechanisms of several feasibility repair methods, including:LOOP-LC, LOOP-LC 2.0. Rather than comparing performances, our goal is to provide an intuitive explanation of different methods.

## Optimization problem

We consider a simple optimization problem:

$$ \text{min } f(p_1,p_2,p_3) \text{  (1a)}$$
s.t. $$ 0\leq p_1\leq 1 \text{  (1b)}$$
$$ 0\leq p_2\leq 1 \text{  (1c)}$$
$$ 0\leq p_3\leq 1 \text{  (1d)}$$
$$ -1\leq p_1+p_2 \leq 1 \text{  (1e)}$$
$$ p_1+p_2+p_3=d_4 \text{  (1f)}$$


where $d_4$ is the input parameter, $p_1$,$p_2$ and $p_3$ are optimization variables. This problem comes from DC optimal power flow problem(see section DCOPF for details) in power system.
Idealy, given $d_4$, the optimization proxy can produce the the optimal solution ${p_1}^*$,${p_2}^*$,${p_3}^*$ to problem (1). However, traditional neural network can't gurantee the feasibility of constraints (1b)~(1f). Therefore, a 'repair layer' can be added to help with feasibility achievement.

We denote the output of the neural network(before repair layer) by $\hat{p_i}$. The following will show how different repair layers work to restore $\hat{p_i}$ to a feasible $\tilde{p_i}$

d--> NN --> $\hat{p_i}$ --> repair layer --> $\tilde{p_i}$

## Feasible Range

To better understand the feasible range of problem (1), let us explore the concept of **variable elimination**. At first glance, problem (1) includes three optimization variables, but only two of them are independent. This is due to the equality constraint (1f), which relates the variables. For instance, if two variables (\(p_1\) and \(p_2\)) are specified, the third variable (\(p_3\)) is determined by \(p_3 = d_4 - p_1 - p_2\). This allows us to reformulate problem (1) as a smaller problem with two independent variables, as shown below:

**Reformulated Problem (2):**

$$ \text{min } f(p_1,p_2,d_4-p_1-p_2) \text{  (2a)}$$
s.t. $$ 0\leq p_1\leq 1 \text{  (2b)}$$
$$ 0\leq p_2\leq 1 \text{  (2c)}$$
$$ 0\leq d_4-p_1-p_2\leq 1 \text{  (2d)}$$
$$ -1\leq p_1+p_2 \leq 1 \text{  (2e)}$$

Once the optimal solution for problem (2) is obtained, the full solution for problem (1) can be derived using \(p_3 = d_4 - p_1 - p_2\).

### Key Insights from Variable Elimination

Variable elimination simplifies the original problem by reducing its dimensionality. Although problem (1) initially appears to involve three variables, only two variables need to be determined. This significantly reduces the computational complexity and aids in visualizing the feasible solution space.

### Visualization of the Feasible Range

The figure below illustrates the feasible range of constraints (1b)-(1f). The blue points represent feasible solutions. Notice that these solutions lie on a 2-dimensional plane within a 3-dimensional space. This confirms that only two variables are independent, consistent with the dimensionality reduction achieved through variable elimination.



