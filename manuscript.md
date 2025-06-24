---
title: 'Context-Adaptive Statistical Inference: Recent Progress, Open Problems, and Opportunities for Foundation Models'
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2025-06-24'
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
  <meta name="dc.title" content="Context-Adaptive Statistical Inference: Recent Progress, Open Problems, and Opportunities for Foundation Models" />
  <meta name="citation_title" content="Context-Adaptive Statistical Inference: Recent Progress, Open Problems, and Opportunities for Foundation Models" />
  <meta property="og:title" content="Context-Adaptive Statistical Inference: Recent Progress, Open Problems, and Opportunities for Foundation Models" />
  <meta property="twitter:title" content="Context-Adaptive Statistical Inference: Recent Progress, Open Problems, and Opportunities for Foundation Models" />
  <meta name="dc.date" content="2025-06-24" />
  <meta name="citation_publication_date" content="2025-06-24" />
  <meta property="article:published_time" content="2025-06-24" />
  <meta name="dc.modified" content="2025-06-24T19:07:25+00:00" />
  <meta property="article:modified_time" content="2025-06-24T19:07:25+00:00" />
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
  <link rel="alternate" type="text/html" href="https://AdaptInfer.github.io/context-review/v/2579a81b2cdb0ca3fa155482999c20915ccc7c9f/" />
  <meta name="manubot_html_url_versioned" content="https://AdaptInfer.github.io/context-review/v/2579a81b2cdb0ca3fa155482999c20915ccc7c9f/" />
  <meta name="manubot_pdf_url_versioned" content="https://AdaptInfer.github.io/context-review/v/2579a81b2cdb0ca3fa155482999c20915ccc7c9f/manuscript.pdf" />
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
([permalink](https://AdaptInfer.github.io/context-review/v/2579a81b2cdb0ca3fa155482999c20915ccc7c9f/))
was automatically generated
from [AdaptInfer/context-review@2579a81](https://github.com/AdaptInfer/context-review/tree/2579a81b2cdb0ca3fa155482999c20915ccc7c9f)
on June 24, 2025.
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

Context-adaptive inference extends classical statistical modeling by allowing parameters to vary across individuals, environments, or tasks. This adaptation may be explicit—through parameterized functions of context—or implicit, via interactions between context and input features. In this review, we survey recent advances in modeling sample-specific variation, including varying-coefficient models, transfer learning, and in-context learning. We also examine the emerging role of foundation models as flexible context encoders. Finally, we outline key challenges and open questions for the development of principled, scalable, and interpretable context-adaptive methods.


## Introduction
A growing number of methods across statistics and machine learning aim to model how data distributions vary across individuals, environments, or tasks. This interest in context-adaptive inference reflects a shift from population-level models toward those that account for sample-specific variation.

In statistics, **varying-coefficient models** allow model parameters to change smoothly with covariates. In machine learning, **meta-learning** and **transfer learning** enable models to adapt across tasks or domains. More recently, **in-context learning** -- by which foundation models adapt behavior based on support examples without parameter updates -- has emerged as a powerful mechanism for personalization in large language models.

These approaches originate from different traditions but share a common goal: to use *context* in the form of covariates, support data, or task descriptors to guide inference about sample-specific *parameters*.

We formalize the setting by assuming each observation $X_i$ is drawn from a sample-specific distribution:

$$X_i \sim P(X; \theta_i)$$

where $\theta_i$ denotes the parameters governing the distribution of the $i$th observation. In the most general case, this formulation allows for arbitrary heterogeneity. However, estimating $N$ distinct parameters from $N$ observations is ill-posed without further structure.

To make the problem tractable, context-adaptive methods introduce structure by assuming that parameters vary systematically with context:
$$
\theta_i = f(c_i).
$$
This deterministic formulation is common in varying-coefficient models and many supervised personalization settings.

More generally, $\theta_i$ may be drawn from a context-dependent distribution:
$$
\theta_i \sim P(\theta \mid c_i),
$$
as in hierarchical Bayesian models or amortized inference frameworks. This stochastic formulation captures residual uncertainty or unmodeled variation beyond what is encoded in $c_i$.

The function $f$ encodes how parameters vary with context, and may be linear, smooth, or nonparametric, depending on the modeling assumptions. In this view, the challenge of context-adaptive inference reduces to estimating or constraining $f$ given data $\{(x_i, c_i)\}_{i=1}^N$.

Viewed this way, context-adaptive inference spans a spectrum—from models that seek **invariance** across environments to models that enable **personalization** at the level of individual samples. For example:

- **Population models** assume $\theta_i = \theta$ for all $i$.
- **Invariant risk minimization** [@doi:10.48550/arXiv.1907.02893] identifies components of $\theta$ that remain stable across distributions.
- **Transfer learning** assumes partial invariance, learning domain-specific shifts around a shared representation.
- **Varying-coefficient models** allow $\theta_i$ to vary smoothly with observed context.
- **In-context learning** treats parameters as an implicit function of support examples.

In this review, we survey methods across this spectrum. We highlight their shared foundations, clarify the assumptions they make about $\theta_i$, and explore the emerging connections between classical approaches such as varying-coefficient models and modern inference mechanisms like in-context learning.

### Population Models
The fundamental assumption of most models is that samples are independent and identically distributed.
However, if samples are identically distributed they must also have identical parameters.
To account for parameter heterogeneity and create more realistic models we must relax this assumption, but the assumption is so fundamental to many methods that alternatives are rarely explored.
Additionally, many traditional models may produce a seemingly acceptable fit to their data, even when the underlying model is heterogeneous.
Here, we explore the consequences of applying homogeneous modeling approaches to heterogeneous data, and discuss how subtle but meaningful effects are often lost to the strength of the identically distributed assumption.

Failure modes of population models can be identified by their error distributions.

__Mode collapse__:
If one population is much larger than another, the other population will be underrepresented in the model.

__Outliers__:
Small populations of outliers can have an enormous effect on OLS models in the parameter-averaging regime.

__Phantom Populations__:
If several populations are present but equally represented, the optimal traditional model will represent none of these populations.

__Lemma:__ A traditional OLS linear model will be the average of heterogeneous models. 

### Context-informed models

Without further assumptions, sample-specific parameter estimation is ill-defined.
Single sample estimation is prohibitively high variance.
We can begin to make this problem tractable by taking note from previous work and imposing assumptions on the topology of $\theta$, or the relationship between $\theta$ and contextual variables.

##### Conditional and Cluster Models
While conditional and cluster models are not truly personalized models, the spirit is the same.
These models make the assumption that models in a single conditional or cluster group are homogeneous. 
More commonly this is written as a group of observations being generated by a single model.
While the assumption results in fewer than $N$ models, it allows the use of generic plug-in estimators.
Conditional or cluster estimators take the form 
$$ \widehat{\theta}_0, ..., \widehat{\theta}_C = \arg\max_{\theta_0, ..., \theta_C} \sum_{c \in \mathcal{C}} \ell(X_c; \theta_c) $$
where $\ell(X; \theta)$ is the log-likelihood of $\theta$ on $X$ and $c$ specifies the covariate group that samples are assigned to, usually by specifying a condition or clustering on covariates thought to affect the distribution of observations. 
Notably, this method produces fewer than $N$ distinct models for $N$ samples and will fail to recover per-sample parameter variation.


##### Distance-regularized Models
Distance-regularized models assume that models with similar covariates have similar parameters and encode this assumption as a regularization term.
$$ \widehat{\theta}_0, ..., \widehat{\theta}_N = \arg\max_{\theta_0, ..., \theta_N} \sum_i \left[ \ell(x_i; \theta_i) \right] - \sum_{i, j} \frac{\| \theta_i - \theta_j \|}{D(c_i, c_j)} $$
The second term is a regularizer that penalizes divergence of $\theta$'s with similar $c$.


##### Parametric Varying-coefficient models
Original paper (based on a smoothing spline function): @doi:10.1111/j.2517-6161.1993.tb01939.x
Markov networks: @doi:10.1080/01621459.2021.2000866
Linear varying-coefficient models assume that parameters vary linearly with covariates, a much stronger assumption than the classic varying-coefficient model but making a conceptual leap that allows us to define a form for the relationship between the parameters and covariates. 
$$\widehat{\theta}_0, ..., \widehat{\theta}_N = \widehat{A} C^T$$
$$ \widehat{A} = \arg\max_A \sum_i \ell(x_i; A c_i) $$


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


## Theoretical Foundations and Advances in Varying-Coefficient Models

### Principles of Adaptivity
TODO: Analyzing the core principles that underpin adaptivity in statistical modeling.

### Advances in Varying-Coefficient Models
TODO: Outlining key theoretical and methodological breakthroughs.

### Integration with State-of-the-Art Machine Learning
TODO: Assessing the enhancement of VC models through modern ML technologies (e.g. deep learning, boosted trees, etc).

### Context-Invariant Training
TODO: The converse of VC models, exploring the implications of training context-invariant models.
e.g. out-of-distribution generalization, robustness to adversarial attacks.

## Context-Adaptive Interpretations of Context-Invariant Models

In the previous section, we discussed the importance of context in model parameters. 
Such context-adaptive models can be learned by explicitly modeling the impact of contextual variables on model parameters, or learned implicitly in a model containing interaction effects between the context and the input features.
In this section, we will focus on recent progress in understanding how context influences interpretations of statistical models, even when the model was not originally designed to incorporate context.

TODO: Discussing the implications of context-adaptive interpretations for traditional models. Related work including LIME/DeepLift/DeepSHAP.


## Opportunities for Foundation Models

### Expanding Frameworks
TODO: Define foundation models, Explore how foundation models are redefining possibilities within statistical models.


### Foundation models as context
TODO: Show recent progress and ongoing directions in using foundation models as context.


## Applications, Case Studies, and Evaluations

### Implementation Across Sectors
TODO: Detailed examination of context-adaptive models in sectors like healthcare and finance.

### Performance Evaluation
TODO: Successes, failures, and comparative analyses of context-adaptive models across applications.


## Technological and Software Tools

### Survey of Tools
TODO: Reviewing current technological supports for context-adaptive models.

### Selection and Usage Guidance
TODO: Offering practical advice on tool selection and use for optimal outcomes.


## Future Trends and Predictions

### Emerging Technologies
TODO: Identifying upcoming technologies and predicting their impact on context-adaptive learning.

### Advances in Methodologies
TODO: Speculating on potential future methodological enhancements.


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

