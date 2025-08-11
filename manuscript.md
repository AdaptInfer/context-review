---
title: 'Context-Adaptive Inference: Bridging Statistical and Foundation Models'
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2025-08-11'
author-meta:
- Ben Lengerich
- Caleb N. Ellington
- Yue Yao
header-includes: |
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta property="og:type" content="article" />
  <meta name="dc.title" content="Context-Adaptive Inference: Bridging Statistical and Foundation Models" />
  <meta name="citation_title" content="Context-Adaptive Inference: Bridging Statistical and Foundation Models" />
  <meta property="og:title" content="Context-Adaptive Inference: Bridging Statistical and Foundation Models" />
  <meta property="twitter:title" content="Context-Adaptive Inference: Bridging Statistical and Foundation Models" />
  <meta name="dc.date" content="2025-08-11" />
  <meta name="citation_publication_date" content="2025-08-11" />
  <meta property="article:published_time" content="2025-08-11" />
  <meta name="dc.modified" content="2025-08-11T16:38:08+00:00" />
  <meta property="article:modified_time" content="2025-08-11T16:38:08+00:00" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Ben Lengerich" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0000-0001-8690-9554" />
  <meta name="twitter:creator" content="@ben_lengerich" />
  <meta name="citation_author" content="Caleb N. Ellington" />
  <meta name="citation_author_institution" content="Computational Biology Department, Carnegie Mellon University" />
  <meta name="citation_author_orcid" content="0000-0001-7029-8023" />
  <meta name="twitter:creator" content="@probablybots" />
  <meta name="citation_author" content="Yue Yao" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0000-8195-3943" />
  <link rel="canonical" href="https://AdaptInfer.github.io/context-review/" />
  <meta property="og:url" content="https://AdaptInfer.github.io/context-review/" />
  <meta property="twitter:url" content="https://AdaptInfer.github.io/context-review/" />
  <meta name="citation_fulltext_html_url" content="https://AdaptInfer.github.io/context-review/" />
  <meta name="citation_pdf_url" content="https://AdaptInfer.github.io/context-review/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://AdaptInfer.github.io/context-review/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://AdaptInfer.github.io/context-review/v/f421426bb929a91b41a1be0179831f4c3761806c/" />
  <meta name="manubot_html_url_versioned" content="https://AdaptInfer.github.io/context-review/v/f421426bb929a91b41a1be0179831f4c3761806c/" />
  <meta name="manubot_pdf_url_versioned" content="https://AdaptInfer.github.io/context-review/v/f421426bb929a91b41a1be0179831f4c3761806c/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://AdaptInfer.github.io/context-review/v/f421426bb929a91b41a1be0179831f4c3761806c/))
was automatically generated
from [AdaptInfer/context-review@f421426](https://github.com/AdaptInfer/context-review/tree/f421426bb929a91b41a1be0179831f4c3761806c)
on August 11, 2025.
</em></small>



## Authors



+ **Ben Lengerich**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-8690-9554](https://orcid.org/0000-0001-8690-9554)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [blengerich](https://github.com/blengerich)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [ben_lengerich](https://twitter.com/ben_lengerich)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
     · Funded by None
  </small>

+ **Caleb N. Ellington**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-7029-8023](https://orcid.org/0000-0001-7029-8023)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [cnellington](https://github.com/cnellington)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [probablybots](https://twitter.com/probablybots)
    <br>
  <small>
     Computational Biology Department, Carnegie Mellon University
     · Funded by None
  </small>

+ **Yue Yao**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0000-8195-3943](https://orcid.org/0009-0000-8195-3943)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [YueYao-stat](https://github.com/YueYao-stat)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
     · Funded by None
  </small>


::: {#correspondence}
✉ — Correspondence possible via [GitHub Issues](https://github.com/AdaptInfer/context-review/issues)

:::


## Abstract {.page_break_before}

Context-adaptive inference enables models to adjust their behavior across individuals, environments, or tasks. 
This adaptivity may be *explicit*, through parameterized functions of context, or *implicit*, as in foundation models that respond to prompts and support in-context learning. 
In this review, we connect recent developments in varying-coefficient models, contextualized learning, and in-context learning. 
We highlight how foundation models can serve as flexible encoders of context, and how statistical methods offer structure and interpretability. 
We propose a unified view of context-adaptive inference and outline open challenges in developing scalable, principled, and personalized models that adapt to the complexities of real-world data.

## Introduction

A convenient simplifying assumption in statistical modeling is that observations are independent and identically distributed (i.i.d.). 
This assumption allows us to use a single model to make predictions across all data points. 
But in practice, this assumption rarely holds. 
Data are collected across different individuals, environments, and tasks -- each with their own characteristics, constraints, and dynamics.

To model this heterogeneity, a growing class of methods aim to make inference *adaptive to context*. These include varying-coefficient models in statistics, transfer and meta-learning in machine learning, and in-context learning in large foundation models. Though these approaches arise from different traditions, they share a common goal: to use contextual information -- whether covariates, environments, or support sets -- to inform sample-specific inference.

We formalize this by assuming each observation $x_i$ is drawn from a distribution governed by parameters $\theta_i$:

$$
x_i \sim P(x; \theta_i).
$$

In population models, the assumption is that $\theta_i = \theta$ for all $i$. In context-adaptive models, we instead posit that the parameters vary with context:

$$
\theta_i = f(c_i) \quad \text{or} \quad \theta_i \sim P(\theta \mid c_i),
$$

where $c_i$ captures the relevant covariates or environment for observation $i$. The goal is to estimate either a deterministic function $f$ or a conditional distribution over parameters.

This shift raises new modeling challenges. 
Estimating a unique $\theta_i$ from a single observation is ill-posed unless we impose structure—smoothness, sparsity, shared representations, or latent grouping. 
And as adaptivity becomes more implicit (e.g., via neural networks or black-box inference), we must develop tools to recover, interpret, or constrain the underlying parameter variation.


In this review, we examine methods that use context to guide inference, either by specifying how parameters change with covariates or by learning to adapt behavior implicitly. 
We begin with classical models that impose explicit structure—such as varying-coefficient models and multi-task learning—and then turn to more flexible approaches like meta-learning and in-context learning with foundation models. 
Though these methods arise from different traditions, they share a common goal: to tailor inference to the local characteristics of each observation or task. 
Along the way, we highlight recurring themes: complex models often decompose into simpler, context-specific components; foundation models can both adapt to and generate context; and context-awareness challenges classical assumptions of homogeneity. 
These perspectives offer a unifying lens on recent advances and open new directions for building adaptive, interpretable, and personalized models.

## From Population Assumptions to Context-Adaptive Inference


Most statistical and machine learning models begin with a foundational assumption: that all samples are drawn independently and identically from a shared population distribution. This assumption simplifies estimation and enables generalization from limited data, but it collapses in the presence of meaningful heterogeneity.

In practice, data often reflect differences across individuals, environments, or conditions. These differences may stem from biological variation, temporal drift, site effects, or shifts in measurement context. Treating heterogeneous data as if it were homogeneous can obscure real effects, inflate variance, and lead to brittle predictions.

### Failure Modes of Population Models

Even when traditional models appear to fit aggregate data well, they may hide systematic failure modes.

**Mode Collapse**  
When one subpopulation is much larger than another, standard models are biased toward the dominant group, underrepresenting the minority group in both fit and predictions.

**Outlier Sensitivity**  
In the parameter-averaging regime, small but extreme groups can disproportionately distort the global model, especially in methods like ordinary least squares.

**Phantom Populations**  
When multiple subpopulations are equally represented, the global model may fit none of them well, instead converging to a solution that represents a non-existent average case.

These behaviors reflect a deeper problem: the assumption of identically distributed samples is not just incorrect, but actively harmful in heterogeneous settings.


### Toward Context-Aware Models

To account for heterogeneity, we must relax the assumption of shared parameters and allow the data-generating process to vary across samples. A general formulation assumes each observation is governed by its own latent parameters:
$$
x_i \sim P(x; \theta_i),
$$

However, estimating $N$ free parameters from $N$ samples is underdetermined. 
Context-aware approaches resolve this by introducing structure on how parameters vary, often by assuming that $\theta_i$ depends on an observed context $c_i$:

$$
\theta_i = f(c_i) \quad \text{or} \quad \theta_i \sim P(\theta \mid c_i).
$$

This formulation makes the model estimable, but it raises new challenges. 
How should $f$ be chosen? How smooth, flexible, or structured should it be? The remainder of this review explores different answers to this question, and shows how implicit and explicit representations of context can lead to powerful, personalized models.

### Early Remedies: Grouped and Distance-Based Models

Before diving into flexible estimators of $f(c)$, we review early modeling strategies that attempt to break away from homogeneity.

#### Conditional and Clustered Models

One approach is to group observations into C contexts, either by manually defining conditions (e.g. male vs. female) or using unsupervised clustering. Each group is then assigned a distinct parameter vector:

$$
\{\widehat{\theta}_0, \ldots, \widehat{\theta}_C\} = \arg\max_{\theta_0, \ldots, \theta_C} \sum_{c \in \mathcal{C}} \ell(X_c; \theta_c),
$$
where $\ell(X; \theta)$ is the log-likelihood of $\theta$ on $X$ and $c$ specifies the covariate group that samples are assigned to. This reduces variance but limits granularity. It assumes that all members of a group share the same distribution and fails to capture variation within a group.

#### Distance-Regularized Estimation

A more flexible alternative assumes that observations with similar contexts should have similar parameters. This is encoded as a regularization penalty that discourages large differences in $\theta_i$ for nearby $c_i$:

$$
\{\widehat{\theta}_0, \ldots, \widehat{\theta}_N\} = \arg\max_{\theta_0, \ldots, \theta_N} \left( \sum_i \ell(x_i; \theta_i) - \sum_{i,j} \frac{\|\theta_i - \theta_j\|}{D(c_i, c_j)} \right),
$$

where $D(c_i, c_j)$ is a distance metric between contexts. This approach allows for smoother parameter variation but requires careful choice of $D$ and regularization strength $\lambda$ to balance bias and variance.  
The choice of distance metric D and regularization strength λ controls the bias–variance tradeoff.

#### Parametric Varying-coefficient models
Original paper (based on a smoothing spline function): @doi:10.1111/j.2517-6161.1993.tb01939.x
Markov networks: @doi:10.1080/01621459.2021.2000866
Linear varying-coefficient models assume that parameters vary linearly with covariates, a much stronger assumption than the classic varying-coefficient model but making a conceptual leap that allows us to define a form for the relationship between the parameters and covariates. 
$$\widehat{\theta}_0, ..., \widehat{\theta}_N = \widehat{A} C^T$$
$$ \widehat{A} = \arg\max_A \sum_i \ell(x_i; A c_i) $$

TODO: Note that they achieve distance-matching by using a distance metric under Euclidean distance, which is a special case of the distance-regularized estimation above.

##### Semi-parametric varying-coefficient Models
Original paper: @doi:10.1214/aos/1017939139
2-step estimation with RBF kernels: @arxiv:2103.00315

Classic varying-coefficient models assume that models with similar covariates have similar parameters, or -- more formally -- that changes in parameters are smooth over the covariate space.
This assumption is encoded as a sample weighting, often using a kernel, where the relevance of a sample to a model is equivalent to its kernel similarity over the covariate space.
$$\widehat{\theta}_0, ..., \widehat{\theta}_N = \arg\max_{\theta_0, ..., \theta_N} \sum_{i, j} \frac{K(c_i, c_j)}{\sum_{k} K(c_i, c_k)} \ell(x_j; \theta_i)$$
This estimator is the simplest to recover $N$ unique parameter estimates. 
However, the assumption here is contradictory to the partition model estimator. 
When the relationship between covariates and parameters is discontinuous or abrupt, this estimator will fail.

##### Contextualized Models
Seminal work @doi:10.48550/arXiv.1705.10301
Contextualized ML generalization and applications: @doi:10.48550/arXiv.2310.11340, @doi:10.48550/arXiv.2111.01104, @doi:10.21105/joss.06469, @doi:10.48550/arXiv.2310.07918, @doi:10.1016/j.jbi.2022.104086, @doi:10.1101/2020.06.25.20140053, @doi:10.1101/2023.12.01.569658, @doi:10.48550/arXiv.2312.14254

Contextualized models make the assumption that parameters are some function of context, but make no assumption on the form of that function. 
In this regime, we seek to estimate the function often using a deep learner (if we have some differentiable proxy for probability):
$$ \widehat{f} = \arg \max_{f \in \mathcal{F}} \sum_i \ell(x_i; f(c_i)) $$

### Latent-structure Models

##### Partition Models
Markov networks: @doi:10.1214/09-AOAS308
Partition models also assume that parameters can be partitioned into homogeneous groups over the covariate space, but make no assumption about where these partitions occur.
This allows the use of information from different groups in estimating a model for a each covariate. 
Partition model estimators are most often utilized to infer abrupt model changes over time and take the form
$$ \widehat{\theta}_0, ..., \widehat{\theta}_N = \arg\max_{\theta_0, ..., \theta_N} \sum_i \ell(x_i; \theta_i) + \sum_{i = 2}^N \text{TV}(\theta_i, \theta_{i-1})$$
Where the regularizaiton term might take the form 
$$\text{TV}(\theta_i, \theta_{i - 1}) = |\theta_i - \theta_{i-1}|$$ 
This still fails to recover a unique parameter estimate for each sample, but gets closer to the spirit of personalized modeling by putting the model likelihood and partition regularizer in competition to find the optimal partitions. 


### Fine-tuned Models and Transfer Learning
Review: @doi:10.48550/arXiv.2206.02058
Noted in foundational literature for linear varying coefficient models @doi:10.1214/aos/1017939139

Estimate a population model, freeze these parameters, and then include a smaller set of personalized parameters to estimate on a smaller subpopulation.
$$ \widehat{\gamma} = \arg\max_{\gamma} = \ell(\gamma; X) $$
$$ \widehat{\theta_c} = \arg\max_{\theta_c} = \ell(\theta_c; \widehat{\gamma}, X_c) $$



### Context-informed and Latent-structure models
Seminal paper: @doi:10.48550/arXiv.1910.06939

Key idea: negative information sharing. Different models should be pushed apart.
$$ \widehat{\theta}_0, ..., \widehat{\theta}_N = \arg\max_{\theta_0, ..., \theta_N, D} \sum_{i=0}^N \prod_{j=0 s.t. D(c_i, c_j) < d}^N P(x_j; \theta_i) P(\theta_i ; \theta_j) $$


### A Spectrum of Context-Awareness

Context-aware models can be viewed along a spectrum of assumptions about the relationship between context and parameters.

**Global models**: $\theta_i = \theta$ for all $i$  
**Grouped models**: $\theta_i = \theta_c$ for some finite set of groups  
**Smooth models**: $\theta_i = f(c_i)$, with $f$ assumed to be continuous or low-complexity  
**Latent models**: $\theta_i \sim P(\theta | c_i)$, with $f$ learned implicitly

Each of these choices encodes different beliefs about how parameters vary. The next section formalizes this variation and examines general principles for adaptivity in statistical modeling.

Relevant references:

- Can Subpopulation Shifts Explain Disagreement in Model Generalization? [@arXiv:2106.04486]




## Principles of Context-Adaptive Inference
What makes a model adaptive? When is it good for a model to be adaptive? While the appeal of adaptivity lies in flexibility and personalized inference, not all adaptivity is good adaptivity. In this section, we formalize the core principles that underlie adaptive modeling.

### 1. Adaptivity requires flexibility
A model cannot adapt unless it has the capacity to represent multiple behaviors. Flexibility may take the form of nonlinearity, hierarchical structure, or modular components that allow different responses in different settings.

- Interaction effects in regression models [@doi:10.1145/2783258.2788613]
- Hierarchical models that allow for varying effects across groups
- Meta-learning and mixtures-of-experts models that learn to adapt based on context
- Varying-coefficient models that allow coefficients to change with context [@doi:10.1111/j.2517-6161.1993.tb01939.x]

### 2. Adaptivity requires a signal of heterogeneity
- Varying-coefficient models adapt parameters based on observed context [@doi:10.1111/j.2517-6161.1993.tb01939.x]
- Contextual bandits adapt actions to context features [@arxiv:1811.04383]
- Multi-domain models adapt across known environments or inferred partitions [@arXiv:2010.07249]

### 3. Modularity improves adaptivity
Adaptive systems are easier to design, debug, and interpret when built from modular parts. Modularity supports targeted adaptation, transferability, and disentanglement.

- []

### 4. Adaptivity implies selectivity
Adaptation must be earned. Overreacting to limited data leads to overfitting. The best adaptive methods include mechanisms for deciding when not to adapt.
- Lepski's method [@arxiv:1508.00249]
- Aggregation of classifiers [@doi:10.1007/978-3-540-45167-9_23]

### 5. Adaptivity is bounded by data efficiency
[@arxiv:1911.12568]

### 6. Adaptivity is not a free lunch

Adaptivity improves performance when heterogeneity is real and informative, but it can degrade performance when variation is spurious. Key tradeoffs include:

- **Bias vs. variance**: More flexible adaptation can reduce bias but increase variance
- **Stability vs. personalization**: Highly adaptive models may overfit to noise or adversarial context
- **Inference cost**: Adaptive inference may be more computationally intensive than global prediction

Understanding these tradeoffs is essential when designing systems for real-world deployment.


### When Adaptivity Fails: Common Failure Modes
Even when all the ingredients are present, adaptivity can backfire. Common failure modes include:

- Spurious Adaptation: Adapting to unstable or confounded features [@arXiv:2010.05761]
- Overfitting in Low-Data Contexts: Attempting fine-grained adaptation with insufficient signal
- Modularity Mis-specification: Adapting in the wrong units or groupings [@arXiv:1911.08731]
- Feedback Loops: Models that change the data distribution they rely on [@doi:10.1145/3097983.3098066]



Related references:




## Explicit Adaptivity: Structured Estimation of $f(c)$

In classical statistical modeling, all observations are typically assumed to share a common set of parameters. However, modern datasets often display significant heterogeneity across individuals, locations, or experimental conditions, making this assumption unrealistic in many real-world applications. To better capture such heterogeneity, recent approaches model parameters as explicit functions of observed context, formalized as $\theta_i = f(c_i)$, where $f$ maps each context to a sample-specific parameter [@hastie1993varying].

This section systematically reviews explicit adaptivity methods, with a focus on structured estimation of $f(c)$. We begin by revisiting classical varying-coefficient models, which provide a conceptual and methodological foundation for modeling context-dependent effects. We then categorize recent advances in explicit adaptivity according to three principal strategies for estimating $f(c)$: (1) smooth nonparametric models that generalize classical techniques, (2) structurally constrained models that incorporate domain-specific knowledge such as spatial or network structure, and (3) learned function approximators that leverage machine learning methods for high-dimensional or complex contexts. Finally, we summarize key theoretical developments and highlight promising directions for future research in this rapidly evolving field.

### Classical Varying-Coefficient Models: A Foundation

Varying-coefficient models (VCMs) are a foundational tool for modeling heterogeneity, as they allow model parameters to vary smoothly with observed context variables [@hastie1993varying; @li2024semi; @botzer2025publication]. In their original formulation, the regression coefficients are treated as nonparametric functions of low-dimensional covariates, such as time or age. The standard VCM takes the form
$$
y_i = \sum_{j=1}^{p} \beta_j(c_i) x_{ij} + \varepsilon_i,
$$
where each $\beta_j(c)$ is an unknown smooth function, typically estimated using kernel smoothing, local polynomials, or penalized splines [@li2024semi].

This approach provides greater flexibility than fixed-coefficient models and is widely used for longitudinal and functional data analysis. The assumption of smoothness makes estimation and theoretical analysis more tractable, but also imposes limitations. Classical VCMs work best when the context is low-dimensional and continuous. They may struggle with abrupt changes, discontinuities, or high-dimensional and structured covariates. In such cases, interpretability and accuracy can be compromised, motivating the development of a variety of modern extensions, which will be discussed in the following sections.

### Advances in Modeling $f(c)$

Recent years have seen substantial progress in the modeling of $f(c)$, the function mapping context to model parameters. These advances can be grouped into three major strategies: (1) **smooth non-parametric models** that extend classical flexibility; (2) **structurally constrained approaches** that encode domain knowledge such as spatial or network topology; and (3) high-capacity **learned function approximators** from machine learning designed for high-dimensional, unstructured contexts. Each strategy addresses specific challenges in modeling heterogeneity, and together they provide a comprehensive toolkit for explicit adaptivity.

#### Smooth Non-parametric Models

This family of models generalizes the classical VCM by expressing $f(c)$ as a flexible, smooth function estimated with basis expansions and regularization. Common approaches include spline-based methods, local polynomial regression, and RKHS-based frameworks. For instance, [@li2024semi] developed a semi-nonparametric VCM using RKHS techniques for imaging genetics, enabling the model to capture complex nonlinear effects. Such methods are central to generalized additive models, supporting both flexibility and interpretability. Theoretical work has shown that penalized splines and kernel methods offer strong statistical guarantees in moderate dimensions, although computational cost and overfitting can become issues as the dimension of $c$ increases.

#### Structurally Constrained Models

Another direction focuses on incorporating structural information into $f(c)$, especially when the context is discrete, clustered, or topologically organized.

**Piecewise-Constant and Partition-Based Models.**
Here, model parameters are allowed to remain constant within specific regions or clusters of the context space, rather than vary smoothly. Approaches include classical grouped estimators and modern partition models, which may learn changepoints using regularization tools like total variation penalties or the fused lasso. This framework is particularly effective for data with abrupt transitions or heterogeneous subgroups.

**Structured Regularization for Spatial, Graph, and Network Data.**
When context exhibits known structure, regularization terms can be designed to promote similarity among neighboring coefficients [@shi2021graph]. For example, spatially varying-coefficient models have been applied to problems in geographical analysis and econometrics, where local effects are expected to vary across adjacent regions [@murakami2025fast; @englert2025spatially; @hu2024varying; @luo2024urban]. On networked data, the network VCM of [@fan2025network] generalizes these ideas by learning both the latent positions and the parameter functions on graphs, allowing the model to accommodate complex relational heterogeneity. Such structural constraints allow models to leverage domain knowledge, improving efficiency and interpretability where smooth models may struggle.

#### Learned Function Approximators

A third class of methods is rooted in modern machine learning, leveraging high-capacity models to approximate $f(c)$ directly from data. These approaches are especially valuable when the context is high-dimensional or unstructured, where classical assumptions may no longer be sufficient.

**Tree-Based Ensembles.**
Gradient boosting decision trees (GBDTs) and related ensemble methods are well suited to tabular and mixed-type contexts. The framework developed by [@wang2025boosted] extends varying-coefficient models by integrating gradient boosting, achieving strong predictive performance with a level of interpretability. These models are typically easier to train and tune than deep neural networks, and their structure lends itself to interpretation with tools such as SHAP.

**Deep Neural Networks.**
For contexts defined by complex, high-dimensional features such as images, text, or sequential data, deep neural networks offer unique advantages for modeling $f(c)$. These architectures can learn adaptive, data-driven representations that capture intricate relationships beyond the scope of classical models. Applications include personalized medicine, natural language processing, and behavioral science, where outcomes may depend on subtle or latent features of the context.

The decision between these machine learning approaches depends on the specific characteristics of the data, the priority placed on interpretability, and computational considerations. Collectively, these advances have significantly broadened the scope of explicit adaptivity, making it feasible to model heterogeneity in ever more complex settings.

### Key Theoretical Advances

The expanding landscape of varying-coefficient models (VCMs) has been supported by substantial theoretical progress, which secures the validity of flexible modeling strategies and guides their practical use. The nature of these theoretical results often reflects the core structural assumptions of each model class.

**Theory for Smooth Non-parametric Models.**
For classical VCMs based on kernel smoothing, local polynomial estimation, or penalized splines, extensive theoretical work has characterized their convergence rates and statistical efficiency. Under standard regularity conditions, these estimators are known to achieve minimax optimality for function estimation in moderate dimensions [@hastie1993varying]. Recent developments, such as the work of [@li2024semi], have established asymptotic normality in semi-nonparametric settings, which enables valid confidence interval construction and hypothesis testing even in complex applications.

**Theory for Structurally Constrained Models.**
When discrete or network structure is incorporated into VCMs, theoretical analysis focuses on identifiability, regularization properties, and conditions for consistent estimation. For example, [@fan2025network] provide non-asymptotic error bounds for estimators in network VCMs, demonstrating that consistency can be attained when the underlying graph topology satisfies certain connectivity properties. In piecewise-constant and partition-based models, results from change-point analysis and total variation regularization guarantee that abrupt parameter changes can be recovered accurately under suitable sparsity and signal strength conditions.

**Theory for High-Capacity and Learned Models.**
The incorporation of machine learning models into VCMs introduces new theoretical challenges. For high-dimensional and sparse settings, oracle inequalities and penalized likelihood theory establish conditions for consistent variable selection and accurate estimation, as seen in methods based on boosting and other regularization techniques [@wang2025boosted; @cheng2020xgboost]. In the context of neural network-based VCMs, the theory is still developing, with current research focused on understanding generalization properties and identifiability in non-convex optimization. This remains an active and important frontier for both statistical and machine learning communities.

These theoretical advances provide a rigorous foundation for explicit adaptivity, ensuring that VCMs can be deployed confidently across a wide range of complex and structured modeling scenarios.

### Synthesis and Future Directions

Selecting an appropriate modeling strategy for $f(c)$ involves weighing flexibility, interpretability, computational cost, and the extent of available domain knowledge. Learned function approximators, such as deep neural networks, offer unmatched capacity for modeling complex, high-dimensional relationships. However, classical smooth models and structurally constrained approaches often provide greater interpretability, transparency, and statistical efficiency. The choice of prior assumptions and the scalability of the estimation procedure are also central considerations in applied contexts.

Looking forward, several trends are shaping the field. One important direction is the integration of varying-coefficient models with foundation models from natural language processing and computer vision. By using pre-trained embeddings as context variables $c_i$, it becomes possible to incorporate large amounts of prior knowledge and extend VCMs to multi-modal and unstructured data sources. Another active area concerns the principled combination of cross-modal contexts, bringing together information from text, images, and structured covariates within a unified VCM framework.

Advances in interpretability and visualization for high-dimensional or black-box coefficient functions are equally important. Developing tools that allow users to understand and trust model outputs is critical for the adoption of VCMs in sensitive areas such as healthcare and policy analysis.

Finally, closing the gap between methodological innovation and practical deployment remains a priority. Although the literature has produced many powerful variants of VCMs, practical adoption is often limited by the availability of software and the clarity of methodological guidance [@botzer2025publication]. Continued investment in user-friendly implementations, open-source libraries, and empirical benchmarks will facilitate broader adoption and greater impact.

In summary, explicit adaptivity through structured estimation of $f(c)$ now forms a core paradigm at the interface of statistical modeling and machine learning. Future progress will focus not only on expanding the expressive power of these models, but also on making them more accessible, interpretable, and practically useful in real-world applications.


## Implicit Adaptivity: Emergent Contextualization in Complex Models

**Introduction: From Explicit to Implicit Adaptivity.**

Traditional models often describe how parameters change by directly specifying a function of context, for example through expressions like $\theta_i = f(c_i)$, where the link between context $c_i$ and parameters $\theta_i$ is fully explicit. In contrast, many modern machine learning systems adapt in fundamentally different ways. Large neural network architectures—particularly foundation models that are now central to state-of-the-art AI research [@bommasani2021opportunities]—show a capacity for adaptation that does not arise from any predefined mapping. Instead, their flexibility emerges naturally from the structure of the model and the breadth of the data seen during training. This phenomenon is known as implicit adaptivity.

Unlike explicit approaches, implicit adaptivity does not depend on directly mapping context to model parameters, nor does it always require context to be formally defined. Such models, by training on large and diverse datasets, internalize broad statistical regularities. As a result, they often display context-sensitive behavior at inference time, even when the notion of context is only implicit or distributed across the input. This capacity for emergent adaptation is especially prominent in foundation models, which can generalize to new tasks and domains without parameter updates, relying solely on the information provided within the input or prompt.

In this section, we offer a systematic review of the mechanisms underlying implicit adaptation. We first discuss the core architectural principles that support context-aware computation in neural networks. Next, we examine how meta-learning frameworks deliberately promote adaptation across diverse tasks. Finally, we focus on the advanced phenomenon of in-context learning in foundation models, which highlights the frontiers of implicit adaptivity in modern machine learning. Through this progression, we aim to clarify the foundations and significance of implicit adaptivity for current and future AI systems.

### Foundations of Implicit Adaptation

The capacity for implicit adaptation does not originate from a single mechanism, but reflects a range of capabilities grounded in fundamental principles of neural network design. Unlike approaches that adjust parameters by directly mapping context to coefficients, implicit adaptation emerges from the way information is processed within a model, even when the global parameters remain fixed. To provide a basis for understanding more advanced forms of adaptation, such as in-context learning, this section reviews the architectural components that enable context-aware computation. We begin with simple context-as-input models and then discuss the more dynamic forms of conditioning enabled by attention mechanisms.

#### Architectural Conditioning via Context Inputs

The simplest form of implicit adaptation appears in neural network models that directly incorporate context as part of their input. In models written as $y_i = g([x_i, c_i]; \Phi)$, context features $c_i$ are concatenated with the primary features $x_i$, and the mapping $g$ is determined by a single set of fixed global weights $\Phi$. Even though these parameters do not change during inference, the network’s nonlinear structure allows it to capture complex interactions. As a result, the relationship between $x_i$ and $y_i$ can vary depending on the specific value of $c_i$.

This basic yet powerful principle is central to many conditional prediction tasks. For example, personalized recommendation systems often combine a user embedding (as context) with item features to predict ratings. Similarly, in multi-task learning frameworks, shared networks learn representations conditioned on task or environment identifiers, which allows a single model to solve multiple related problems [@ruder2017overview].

#### Interaction Effects and Attention Mechanisms

Modern architectures go beyond simple input concatenation by introducing interaction layers that support richer context dependence. These can include feature-wise multiplications, gating modules, or context-dependent normalization. Among these innovations, the attention mechanism stands out as the foundation of the Transformer architecture [@vaswani2017attention].

Attention allows a model to assign varying degrees of importance to different parts of an input sequence, depending on the overall context. In the self-attention mechanism, each element in a sequence computes a set of query, key, and value vectors. The model then evaluates the relevance of each element to every other element, and these relevance scores determine a weighted sum of the value vectors. This process enables the model to focus on the most relevant contextual information for each step in computation. The ability to adapt processing dynamically in this way is not dictated by explicit parameter functions, but emerges from the network’s internal organization. Such mechanisms make possible the complex forms of adaptation observed in large language models and set the stage for advanced phenomena like in-context learning.

### Amortized Inference and Meta-Learning

Moving beyond fixed architectures that implicitly adapt, another family of methods deliberately trains models to become efficient learners. These approaches, broadly termed meta-learning or "learning to learn," distribute the cost of adaptation across a diverse training phase. As a result, models can make rapid, task-specific adjustments during inference. Rather than focusing on solving a single problem, these methods train models to learn the process of problem-solving itself. This perspective provides an important conceptual foundation for understanding the in-context learning capabilities of foundation models.

#### Amortized Inference

Amortized inference represents a more systematic form of implicit adaptation. In this setting, a model learns a reusable function that enables rapid inference for new data points, effectively distributing the computational cost over the training phase. In traditional Bayesian inference, calculating the posterior distribution for each new data point is computationally demanding. Amortized inference addresses this challenge by training an "inference network" to approximate these calculations. A classic example is the encoder in a Variational Autoencoder (VAE), which is optimized to map high-dimensional observations directly to the parameters, such as mean and variance, of an approximate posterior distribution over a latent space [@kingma2013auto]. The inference network thus learns a complex, black-box mapping from the data context to distributional parameters. Once learned, this mapping can be efficiently applied to any new input at test time, providing a fast feed-forward approximation to a traditionally costly inference process.

#### Meta-Learning: Learning to Learn

Meta-learning builds upon these ideas by training models on a broad distribution of related tasks. The explicit goal is to enable efficient adaptation to new tasks. Instead of optimizing performance for any single task, meta-learning focuses on developing a transferable adaptation strategy or a parameter initialization that supports rapid learning in novel settings [@hospedales2021meta].

Gradient-based meta-learning frameworks such as Model-Agnostic Meta-Learning (MAML) illustrate this principle. In these frameworks, the model discovers a set of initial parameters that can be quickly adapted to a new task with only a small number of gradient updates [@finn2017model]. Training proceeds in a nested loop: the inner loop simulates adaptation to individual tasks, while the outer loop updates the initial parameters to improve adaptability across tasks. As a result, the capacity for adaptation becomes encoded in the meta-learned parameters themselves. When confronted with a new task at inference, the model can rapidly achieve strong performance using just a few examples, without the need for a hand-crafted mapping from context to parameters. This stands in clear contrast to explicit approaches, which rely on constructing and estimating a direct mapping from context to model coefficients.

### In-Context Learning in Foundation Models

The most powerful and, arguably, most enigmatic form of implicit adaptivity is **in-context learning (ICL)**, an emergent capability of large-scale foundation models. This phenomenon has become a central focus of modern AI research, as it represents a significant shift in how models learn and adapt to new tasks. This section provides an expanded review of ICL, beginning with a description of the core phenomenon, then deconstructing the key factors that influence its performance, reviewing the leading hypotheses for its underlying mechanisms, and concluding with its current limitations and open questions.

#### The Phenomenon of Few-Shot In-Context Learning

First systematically demonstrated in large language models such as GPT-3 [@brown2020language], ICL is the ability of a model to perform a new task after being conditioned on just a few examples provided in its input prompt. Critically, this adaptation occurs entirely within a single forward pass, without any updates to the model's weights. For instance, a model can be prompted with a few English-to-French translation pairs and then successfully translate a new word, effectively learning the task on the fly. This capability supports a broad range of applications, including few-shot classification, following complex instructions, and even inducing and applying simple algorithms from examples.

#### Deconstructing ICL: Key Influencing Factors

**The Role of Scale.**
A critical finding is that ICL is an *emergent ability* that appears only after a model surpasses a certain threshold in scale (in terms of parameters, data, and computation). Recent work has shown that larger models do not just improve quantitatively at ICL; they may also learn in qualitatively different ways, suggesting that scale enables a fundamental shift in capability rather than a simple performance boost [@wei2022emergent].

**Prompt Engineering and Example Selection.**
The performance of ICL is highly sensitive to the composition of the prompt. The format, order, and selection of the in-context examples can dramatically affect the model's output. Counterintuitively, research has shown that the distribution of the input examples, rather than the correctness of their labels, often matters more for effective ICL. This suggests that the model is primarily learning a task format or an input-output mapping from the provided examples, rather than learning the underlying concepts from the labels themselves [@min2022rethinking].

#### Hypothesized Mechanisms: How Does ICL Work?

The underlying mechanisms that enable ICL are not fully understood and remain an active area of research. Several leading hypotheses have emerged, viewing ICL through the lenses of meta-learning, Bayesian inference, and specific architectural components.

**ICL as Implicit Meta-Learning.**
The most prominent theory posits that transformers learn to implement general-purpose learning algorithms within their forward pass. During pre-training on vast and diverse datasets, the model is exposed to a multitude of tasks and patterns. This process is thought to implicitly train the model as a meta-learner, allowing it to recognize abstract task structures within a prompt and then execute a learned optimization process on the provided examples to solve the task for a new query [@dai2022can; @von2023transformers].

**ICL as Implicit Bayesian Inference.**
A complementary and powerful perspective understands ICL as a form of implicit Bayesian inference. In this view, the model learns a broad prior over a large class of functions during its pre-training phase. The in-context examples provided in the prompt act as evidence, which the model uses to perform a Bayesian update, resulting in a posterior predictive distribution for the final query. This framework provides a compelling explanation for how models can generalize from very few examples [@xie2021an].

**The Role of Induction Heads.**
From a more mechanistic, architectural perspective, researchers have identified specific attention head patterns, dubbed "induction heads," that appear to be crucial for ICL. These specialized heads are hypothesized to form circuits that can scan the context for repeated patterns and then copy or complete them, providing a basic mechanism for pattern completion and generalization from in-context examples [@olsson2022incontext].

#### Limitations and Open Questions

Despite its remarkable capabilities, ICL faces significant limitations with respect to transparency, explicit control, and robustness. The adaptation process is opaque, making it difficult to debug or predict failure modes. Furthermore, performance can be brittle and highly sensitive to small changes in the prompt. As summarized in recent surveys, key open questions include developing a more complete theoretical understanding of ICL, improving its reliability, and establishing methods for controlling its behavior in high-stakes applications [@dong2022survey].

### Comparative Synthesis: Implicit versus Explicit Adaptivity

Implicit and explicit adaptation strategies represent two fundamentally different philosophies for modeling heterogeneity, each with distinct strengths and limitations. The optimal choice between these approaches depends on the goals of analysis, the structure and scale of available data, and the need for interpretability or regulatory compliance in the application domain.

* **Implicit Adaptivity**: This strategy offers exceptional flexibility and scalability, making it well suited for high-dimensional or unstructured data and efficient at inference. However, the adaptation mechanisms are typically opaque, making it challenging to interpret or control the model’s decision process. In applications like healthcare or autonomous systems, this lack of transparency can hinder trust, validation, and responsible deployment.

* **Explicit Adaptivity**: In contrast, explicit models provide direct, interpretable mappings from context to parameters through functions such as $f(c)$. This structure supports clear visualization, statistical analysis, and the formulation of scientific hypotheses. It also enables more direct scrutiny and control of the model’s reasoning. Nevertheless, explicit methods rely heavily on domain expertise to specify an appropriate functional form, and may struggle to accommodate unstructured or highly complex context spaces. If the assumed structure is misspecified, the model’s performance and generalizability can be severely limited.

In summary, these two paradigms illustrate a fundamental trade-off between expressive capacity and transparent reasoning. Practitioners should carefully weigh these considerations, often choosing or blending approaches based on the unique demands of the task. For clarity, a comparative table or figure can further highlight the strengths and limitations of each strategy across various real-world applications.

### Open Challenges and the Motivation for Interpretability

The rise of powerful implicit adaptation methods, particularly in-context learning, raises critical open research questions regarding their diagnosis, control, and reliability. As these models are deployed in increasingly high-stakes applications, understanding their failure modes is not just an academic exercise but a practical necessity [@bommasani2021opportunities]. It is important to develop systematic methods for assessing when and why in-context learning is likely to fail, and to create techniques for interpreting and, where possible, steering the adaptation process. While direct control remains elusive, recent prompting techniques like Chain-of-Thought suggest that structuring the context can guide the model's internal reasoning process, offering a limited but important form of behavioral control [@wei2022chain]. A thorough understanding of the theoretical limits and practical capabilities of implicit adaptivity remains a central topic for ongoing research.

These considerations motivate a growing search for techniques that can make the adaptation process more transparent by "making the implicit explicit." Such methods aim to bridge the gap between the powerful but opaque capabilities of implicit models and the need for trustworthy, reliable AI. This research can be broadly categorized into several areas, including post-hoc interpretability approaches that seek to explain individual predictions [@linardatos2020explainable], surrogate modeling where a simpler, interpretable model is trained to mimic the complex model's behavior, and strategies for extracting modular structure from trained models. A prime example of the latter is the line of work probing language models to determine if they have learned factual knowledge in a structured, accessible way [@petroni2019language]. By surfacing the latent structure inside these systems, researchers can enhance trust, promote modularity, and improve the readiness of adaptive models for deployment in real-world settings. This line of work provides a conceptual transition to subsequent sections, which explore the integration of interpretability with adaptive modeling.


## Making Implicit Adaptivity Explicit: Local Models, Surrogates and Post Hoc Approximations

This section focuses on methods that aim to extract, approximate, or control the internal adaptivity mechanisms of black-box models. These approaches recognize that implicit adaptivity—while powerful—can be opaque, hard to debug, and brittle to distribution shift. By surfacing structure, we gain interpretability, composability, and sometimes improved generalization.

### Motivation

- Implicit adaptivity can succeed without explicit modeling, but:
  - It obscures *why* and *how* a model adapts
  - It limits modular reuse and inspection
  - It makes personalization hard to constrain or audit
- Making adaptivity explicit supports:
  - Better alignment with downstream goals
  - Composability of learned modules
  - Debugging and error attribution

### Approaches

#### Surrogate Modeling

- Fit interpretable surrogates (e.g., linear models, decision trees) to approximate model behavior locally
- Applications:
  - Explaining predictions post-hoc
  - Approximating $f(c)$ from input-output behavior
- References:
  - LIME, SHAP

#### Prototype and Nearest-Neighbor Methods

- Use nearest neighbors in representation space to approximate model adaptation
- Enables interpretability and modular updates
- Related to contextual bandits, exemplar models

#### Amortization Diagnostics

- For amortized inference (e.g., variational autoencoders), analyze encoder mappings to understand how $q(\theta | x)$ varies with $x$
- Could treat encoder as a learned $f(c)$ and evaluate its fidelity

#### Disentangled Representations

- Train models with constraints (e.g., variational regularization, info bottlenecks) to encourage explicit factors of variation
- Goal: make parameter changes traceable to distinct contextual causes

#### Parameter Extraction

- Techniques like linear probes, weight attribution, or synthetic tasks to reverse-engineer how models adapt internally
- Example: "what part of the weights encode the task?"

### Tradeoffs

- Fidelity vs interpretability
- Local vs global explanations
- Approximation error vs modular control

### Open Questions

- Can we extract *portable* modules from foundation models?
- When does making structure explicit improve performance?
- What is the right level of abstraction—parameters, functions, latent causes?

This section bridges black-box adaptation and structured inference. It highlights how interpretability and performance need not be at odds—especially when the goal is robust, composable, and trustworthy adaptation.

TODO: Discussing the implications of context-adaptive interpretations for traditional models. Related work including LIME/DeepLift/DeepSHAP.

Relevant references:

- [@arxiv:2310.05797]
- Interpretations are statistics [@arXiv:2402.02870]


## Context-Invariant Training: A View from the Converse
TODO: The converse of context-adaptive models, exploring the implications of training context-invariant models.
e.g. out-of-distribution generalization, robustness to adversarial attacks.

Relevant references:

- Invariant Risk Minimization [@arXiv:1907.02893]
- Out-of-Distribution Generalization via Risk Extrapolation [@arXiv:2003.00688]
- The Risks of Invariant Risk Minimization [@arXiv:2010.05761]
- Conditional Variance Penalties and Domain Adaptation [@arXiv:1710.11469]
- Can Subpopulation Shifts Explain Disagreement in Model Generalization? [@arXiv:2106.04486]

### Adversarial Robustness as Context-Invariant Training
Related references:

- Towards Deep Learning Models Resistant to Adversarial Attacks [@arXiv:1706.06083]
- Robustness May Be at Odds with Accuracy [@arXiv:1805.12152]

### Training methods for Context-Invariant Models
- Just Train Twice: Improving Group Robustness without Training Group Information [@arXiv:2002.10384]
- Environment Inference for Invariant Learning [@arXiv:2110.14048]
- Distributionally Robust Neural Networks for Group Shifts [@arXiv:1911.08731]



## Applications, Case Studies, Evaluation Metrics, and Tools

### Implementation Across Sectors
TODO: Detailed examination of context-adaptive models in sectors like healthcare and finance.

Relevant references:

- [@doi:10.6339/25-JDS1181]
- [@doi:10.3390/math13030469]

### Performance Evaluation
TODO: Successes, failures, and comparative analyses of context-adaptive models across applications.


### Survey of Tools
TODO: Reviewing current technological supports for context-adaptive models.

### Selection and Usage Guidance
TODO: Offering practical advice on tool selection and use for optimal outcomes.


## Future Trends and Opportunities with Foundation Models

### Emerging Technologies
TODO: Identifying upcoming technologies and predicting their impact on context-adaptive learning.

### Advances in Methodologies
TODO: Speculating on potential future methodological enhancements.


### Expanding Frameworks with Foundation Models

Foundation models refer to large-scale, general-purpose neural networks, predominantly transformer-based architectures, trained on vast datasets using self-supervised learning [@doi:10.48550/arXiv.2108.07258]. These models have significantly transformed modern statistical modeling and machine learning due to their flexibility, adaptability, and strong performance across diverse domains. Notably, large language models (LLMs) such as GPT-4 [@doi:10.48550/arXiv.2303.08774] and LLaMA-3.1 [@doi:10.48550/arXiv.2407.21783] have achieved substantial advancements in natural language processing (NLP), demonstrating proficiency in tasks ranging from text generation and summarization to question-answering and dialogue systems. Beyond NLP, foundation models also excel in multimodal (text-vision) tasks [@doi:10.48550/arXiv.2103.00020], text embedding generation [@doi:10.48550/arXiv.1810.04805], and structured tabular data analysis [@doi:10.48550/arXiv.2207.01848], highlighting their broad applicability.

A key strength of foundation models lies in their capacity to dynamically adapt to different contexts provided by inputs. This adaptability is primarily achieved through techniques such as prompting, which involves designing queries to guide the model's behavior implicitly, allowing task-specific responses without additional fine-tuning [@doi:10.1145/3560815]. Furthermore, mixture-of-experts (MoE) architectures amplify this contextual adaptability by employing routing mechanisms that select specialized sub-models or "experts" tailored to specific input data, thus optimizing computational efficiency and performance [@doi:10.1007/s10462-012-9338-y].

#### **Foundation Models as Context**

Foundation models offer significant opportunities by supplying context-aware information that enhances various stages of statistical modeling and inference:

**Feature Extraction and Interpretation:** Foundation models transform raw, unstructured data into structured and interpretable representations. For example, targeted prompts enable LLMs to extract insightful features from text, providing meaningful insights and facilitating interpretability [@doi:10.48550/arXiv.2302.12343, @doi:10.48550/arXiv.2305.12696, @doi:10.18653/v1/2023.emnlp-main.384]. This allows statistical models to operate directly on semantically meaningful features rather than on raw, less interpretable data.

**Contextualized Representations for Downstream Modeling:** Foundation models produce adaptable embeddings and intermediate representations useful as inputs for downstream models, such as decision trees or linear models [@doi:10.48550/arXiv.2208.01066]. These embeddings significantly enhance the training of both complex, black-box models [@doi:10.48550/arXiv.2212.09741] and simpler statistical methods like n-gram-based analyses [@doi:10.1038/s41467-023-43713-1], thereby broadening the application scope and effectiveness of statistical approaches.

**Post-hoc Interpretability:** Foundation models support interpretability by generating natural-language explanations for decisions made by complex models. This capability enhances transparency and trust in statistical inference, providing clear insights into how and why certain predictions or decisions are made [@doi:10.48550/arXiv.2409.08466].

Recent innovations underscore the role of foundation models in context-sensitive inference and enhanced interpretability:

**FLAN-MoE** (Fine-tuned Language Model with Mixture of Experts) [@doi:10.48550/arXiv.2305.14705] combines instruction tuning with expert selection, dynamically activating relevant sub-models based on the context. This method significantly improves performance across diverse NLP tasks, offering superior few-shot and zero-shot capabilities. It also facilitates interpretability through explicit expert activations. Future directions may explore advanced expert-selection techniques and multilingual capabilities.

**LMPriors** (Pre-Trained Language Models as Task-Specific Priors) [@doi:10.48550/arXiv.2210.12530] leverages semantic insights from pre-trained models like GPT-3 to guide tasks such as causal inference, feature selection, and reinforcement learning. This method markedly enhances decision accuracy and efficiency without requiring extensive supervised datasets. However, it necessitates careful prompt engineering to mitigate biases and ethical concerns.

**Mixture of In-Context Experts** (MoICE) [@doi:10.48550/arXiv.2210.12530] introduces a dynamic routing mechanism within attention heads, utilizing multiple Rotary Position Embeddings (RoPE) angles to effectively capture token positions in sequences. MoICE significantly enhances performance on long-context sequences and retrieval-augmented generation tasks by ensuring complete contextual coverage. Efficiency is achieved through selective router training, and interpretability is improved by explicitly visualizing attention distributions, providing detailed insights into the model's reasoning process.


## Open Problems

### Theoretical Challenges
TODO: Critically examining unresolved theoretical issues like identifiability, etc.

### Ethical and Regulatory Considerations
TODO: Discussing the ethical landscape and regulatory challenges, with focus on benefits of interpretability and regulatability.

### Complexity in Implementation
TODO: Addressing obstacles in practical applications and gathering insights from real-world data.

TODO: Other open problems?

## Conclusion

### Overview of Insights
TODO: Summarizing the main findings and contributions of this review.


### Future Directions
TODO: Discussing potential developments and innovations in context-adaptive statistical inference.

## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

