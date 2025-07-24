---
title: 'Context-Adaptive Inference: Bridging Statistical and Foundation Models'
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2025-07-24'
author-meta:
- Ben Lengerich
- Caleb N. Ellington
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
  <meta name="dc.date" content="2025-07-24" />
  <meta name="citation_publication_date" content="2025-07-24" />
  <meta property="article:published_time" content="2025-07-24" />
  <meta name="dc.modified" content="2025-07-24T16:57:11+00:00" />
  <meta property="article:modified_time" content="2025-07-24T16:57:11+00:00" />
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
  <link rel="canonical" href="https://AdaptInfer.github.io/context-review/" />
  <meta property="og:url" content="https://AdaptInfer.github.io/context-review/" />
  <meta property="twitter:url" content="https://AdaptInfer.github.io/context-review/" />
  <meta name="citation_fulltext_html_url" content="https://AdaptInfer.github.io/context-review/" />
  <meta name="citation_pdf_url" content="https://AdaptInfer.github.io/context-review/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://AdaptInfer.github.io/context-review/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://AdaptInfer.github.io/context-review/v/9aee3ed0e8253e56502960bad5a3aecada47138a/" />
  <meta name="manubot_html_url_versioned" content="https://AdaptInfer.github.io/context-review/v/9aee3ed0e8253e56502960bad5a3aecada47138a/" />
  <meta name="manubot_pdf_url_versioned" content="https://AdaptInfer.github.io/context-review/v/9aee3ed0e8253e56502960bad5a3aecada47138a/manuscript.pdf" />
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
([permalink](https://AdaptInfer.github.io/context-review/v/9aee3ed0e8253e56502960bad5a3aecada47138a/))
was automatically generated
from [AdaptInfer/context-review@9aee3ed](https://github.com/AdaptInfer/context-review/tree/9aee3ed0e8253e56502960bad5a3aecada47138a)
on July 24, 2025.
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

TODO: Sync with overview.md

### Varying-Coefficient Models

### Recent Advances in Varying-Coefficient Models
TODO: Outlining key theoretical and methodological breakthroughs.

Relevant references:

- [@doi:10.3390/publications13020019]

#### Flexible Functional Forms

Relevant references:

- [@doi:10.5705/ss.202024.0118]

#### Integration with State-of-the-Art Machine Learning
TODO: Enhancing VC models with modern ML technologies (e.g. deep learning, boosted trees, etc).

Relevant references:

- [@doi:10.1007/s00180-025-01603-8]
- [@arxiv:2003.06416]
- [@arxiv:2004.13912]

#### Structured data (Spatio-Temporal, Graphs, etc.)

Related references:

- [@doi:10.1080/01621459.2025.2470481]
- [@arxiv:2502.14651]
- [@doi:10.1111/gean.70005]
- [@doi:10.1016/j.jeconom.2024.105883]
- [@doi:10.1016/j.regsciurbeco.2024.104009]
- [@arXiv:2111.01104]



## Implicit Adaptivity: Emergent Contextualization within Complex Models


Not all models adapt through explicit parameterization. In many modern systems, adaptation emerges from architecture, training data, or inference dynamics—without being hard-coded as a function of context.

We refer to this as *implicit adaptivity*. These methods do not model $\theta_i$ directly as a function of $c_i$, nor do they always define context formally. Instead, they internalize patterns across training distributions in a way that enables flexible behavior at inference time.

A canonical example is **in-context learning** with foundation models. Given a prompt consisting of a few examples, the model adjusts its behavior—often achieving personalization or task adaptation—without updating weights or making any explicit inference over $\theta$. This capacity arises from pretraining on diverse data and from the model’s architecture, not from structured estimation.

Other forms of implicit adaptivity include:

- **Fine-tuned models** that generalize across tasks or domains by adjusting shared components.
- **Attention-based architectures** that condition on context without defining a parametric mapping.
- **Gradient-based meta-learners** trained to produce fast adaptation without modeling $\theta(c)$ explicitly.

These methods challenge the boundary between training and inference. They blur the distinction between model parameters and data inputs, and they rely on massive-scale training to amortize the cost of adaptation.

In this section, we examine:

- How implicit adaptivity arises in foundation models
- What assumptions these models make (implicitly or explicitly) about context
- How their performance compares to structured, explicit approaches
- When it’s valuable to make the adaptation process more interpretable or modular

Implicit adaptivity offers powerful capabilities, but it also hides structure that could be useful for analysis, debugging, or control. The next section explores efforts to *make the implicit explicit*—by approximating, interpreting, or extracting the latent adaptation mechanisms inside black-box models.

### Defining Implicit Adaptation

### Neural Networks with context inputs (e.g. interaction effects, attention mechanisms, etc.)

### Amortized Inference and Meta-Learning

### In-context learning in transformers and foundation models


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

