---
title: 'Context-Adaptive Inference: Bridging Statistical and Foundation Models'
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2025-10-17'
author-meta:
- Yue Yao
- Caleb N. Ellington
- Jingyun Jia
- Baiheng Chen
- Dong Liu
- Rikhil Rao
- Jiaqi Wang
- Samuel Wales-McGrath
- Yixin Yang
- Zhiyuan Li
- Ben Lengerich
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
  <meta name="dc.date" content="2025-10-17" />
  <meta name="citation_publication_date" content="2025-10-17" />
  <meta property="article:published_time" content="2025-10-17" />
  <meta name="dc.modified" content="2025-10-17T22:09:51+00:00" />
  <meta property="article:modified_time" content="2025-10-17T22:09:51+00:00" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Yue Yao" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0000-8195-3943" />
  <meta name="citation_author" content="Caleb N. Ellington" />
  <meta name="citation_author_institution" content="Computational Biology Department, Carnegie Mellon University" />
  <meta name="citation_author_orcid" content="0000-0001-7029-8023" />
  <meta name="twitter:creator" content="@probablybots" />
  <meta name="citation_author" content="Jingyun Jia" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0006-3241-3485" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Baiheng Chen" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0000-0001-8554-3984" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Dong Liu" />
  <meta name="citation_author_institution" content="Department of Computer Science, Yale University" />
  <meta name="citation_author_orcid" content="0009-0009-6815-8297" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Rikhil Rao" />
  <meta name="citation_author_institution" content="Department of Computer Science, University of Wisconsin - Madison" />
  <meta name="citation_author_orcid" content="0009-0003-5221-1125" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Jiaqi Wang" />
  <meta name="citation_author_institution" content="Paul G. Allen School of Computer Science &amp; Engineering, University of Washington" />
  <meta name="citation_author_orcid" content="0009-0003-8531-9490" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Samuel Wales-McGrath" />
  <meta name="citation_author_institution" content="Department of Computer Science and Engineering, The Ohio State University" />
  <meta name="citation_author_orcid" content="0009-0008-5405-2646" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Yixin Yang" />
  <meta name="citation_author_institution" content="Department of Computer Sciences, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0008-1436-5584" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Zhiyuan Li" />
  <meta name="citation_author_institution" content="Department of Computer Sciences, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0006-6016-7381" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Ben Lengerich" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0000-0001-8690-9554" />
  <meta name="twitter:creator" content="@ben_lengerich" />
  <link rel="canonical" href="https://AdaptInfer.github.io/context-review/" />
  <meta property="og:url" content="https://AdaptInfer.github.io/context-review/" />
  <meta property="twitter:url" content="https://AdaptInfer.github.io/context-review/" />
  <meta name="citation_fulltext_html_url" content="https://AdaptInfer.github.io/context-review/" />
  <meta name="citation_pdf_url" content="https://AdaptInfer.github.io/context-review/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://AdaptInfer.github.io/context-review/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://AdaptInfer.github.io/context-review/v/ac66b0d295e296c28aab59127de1980357aaaf33/" />
  <meta name="manubot_html_url_versioned" content="https://AdaptInfer.github.io/context-review/v/ac66b0d295e296c28aab59127de1980357aaaf33/" />
  <meta name="manubot_pdf_url_versioned" content="https://AdaptInfer.github.io/context-review/v/ac66b0d295e296c28aab59127de1980357aaaf33/manuscript.pdf" />
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
([permalink](https://AdaptInfer.github.io/context-review/v/ac66b0d295e296c28aab59127de1980357aaaf33/))
was automatically generated
from [AdaptInfer/context-review@ac66b0d](https://github.com/AdaptInfer/context-review/tree/ac66b0d295e296c28aab59127de1980357aaaf33)
on October 17, 2025.
</em></small>



## Authors



+ **Yue Yao**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0000-8195-3943](https://orcid.org/0009-0000-8195-3943)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [YueYao-stat](https://github.com/YueYao-stat)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
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
  </small>

+ **Jingyun Jia**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0006-3241-3485](https://orcid.org/0009-0006-3241-3485)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [Clouddelta](https://github.com/Clouddelta)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
  </small>

+ **Baiheng Chen**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-8554-3984](https://orcid.org/0000-0001-8554-3984)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [BaihengChen](https://github.com/BaihengChen)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
  </small>

+ **Dong Liu**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0009-6815-8297](https://orcid.org/0009-0009-6815-8297)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [NoakLiu](https://github.com/NoakLiu)
    <br>
  <small>
     Department of Computer Science, Yale University
  </small>

+ **Rikhil Rao**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0003-5221-1125](https://orcid.org/0009-0003-5221-1125)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [rikhilr](https://github.com/rikhilr)
    <br>
  <small>
     Department of Computer Science, University of Wisconsin - Madison
  </small>

+ **Jiaqi Wang**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0003-8531-9490](https://orcid.org/0009-0003-8531-9490)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [w-jiaqi](https://github.com/w-jiaqi)
    <br>
  <small>
     Paul G. Allen School of Computer Science & Engineering, University of Washington
  </small>

+ **Samuel Wales-McGrath**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0008-5405-2646](https://orcid.org/0009-0008-5405-2646)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [Samuel-WM](https://github.com/Samuel-WM)
    <br>
  <small>
     Department of Computer Science and Engineering, The Ohio State University
  </small>

+ **Yixin Yang**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0008-1436-5584](https://orcid.org/0009-0008-1436-5584)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [Jessego5](https://github.com/Jessego5)
    <br>
  <small>
     Department of Computer Sciences, University of Wisconsin-Madison
  </small>

+ **Zhiyuan Li**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0006-6016-7381](https://orcid.org/0009-0006-6016-7381)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [LZYEIL](https://github.com/LZYEIL)
    <br>
  <small>
     Department of Computer Sciences, University of Wisconsin-Madison
  </small>

+ **Ben Lengerich**
  ^[✉](#correspondence)^<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-8690-9554](https://orcid.org/0000-0001-8690-9554)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [blengerich](https://github.com/blengerich)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [ben_lengerich](https://twitter.com/ben_lengerich)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
  </small>


::: {#correspondence}
✉ — Correspondence possible via [GitHub Issues](https://github.com/AdaptInfer/context-review/issues)
or email to
Ben Lengerich \<lengerich@wisc.edu\>.


:::


## Abstract {.page_break_before}

Context-adaptive inference enables models to adjust their behavior across individuals, environments, or tasks. 
Adaptation may be *explicit*, through parameterized functions of context, or *implicit*, as in foundation models that respond to prompts and support in-context learning. 
This review synthesizes advances in varying coefficient models and adaptive behavior in foundation models, and discusses how statistical and neural approaches converge in context-aware inference. 
We highlight how foundation models can serve as flexible encoders of context, and how statistical methods offer structure and interpretability. 
Building on these links, we propose a common perspective that places explicit and implicit mechanisms on the same spectrum of context use. 
We also identify open problems in identifiability, robustness under distribution shift, and efficient large-scale adaptation, outlining design principles for methods that are scalable, reliable, and transparent in real-world settings.


## Introduction

A convenient simplifying assumption in statistical modeling is that observations are independent and identically distributed (i.i.d.). This assumption allows us to use a single model to make predictions across all data points. But in practice, this assumption rarely holds. Data are collected across different individuals, environments, and tasks—each with their own characteristics, constraints, and dynamics. When the i.i.d. assumption breaks down, using a single global model can obscure meaningful heterogeneity.

To model this heterogeneity, a growing class of methods aim to make inference *adaptive to context*. These include varying-coefficient models in statistics, transfer and meta-learning in machine learning, and in-context learning in large foundation models. Though these approaches arise from different traditions, they share a common goal: to use contextual information—whether covariates, environments, or support sets—to inform sample-specific inference.

We formalize this by assuming each observation $x_i$ is drawn from a distribution governed by parameters $\theta_i$:
$$
x_i \sim P(x;\,\theta_i).
$$

In population models, the assumption is that $\theta_i=\theta$ for all $i$. In context-adaptive models, we instead posit that the parameters vary with context:
$$
\theta_i=f(c_i)\quad\text{or}\quad \theta_i\sim P(\theta\mid c_i),
$$
where $c_i$ captures the relevant covariates or environment for observation $i$. The goal is to estimate either a deterministic function $f$ or a conditional distribution over parameters.

This shift raises new modeling challenges. Estimating a unique $\theta_i$ from a single observation is ill-posed without structural regularization—smoothness, sparsity, shared representations, or latent grouping. And as adaptivity becomes more implicit (e.g., via neural networks or black-box inference), we need tools to recover, interpret, or constrain the underlying parameter variation.

### Problem Setup and Notation

We study supervised prediction with units $i=1,\dots,n$. Each unit has a **context** $c_i\in\mathcal{C}$ (e.g., patient/user/site/time) and observed data $\mathcal{D}_i=\{(x_{ij},y_{ij})\}_{j=1}^{m_i}$ with $x_{ij}\in\mathcal{X}$ and $y_{ij}\in\mathcal{Y}$. Predictions come from a model family $\mathcal{H}=\{h_\theta:\mathcal{X}\to\mathcal{Y}\mid \theta\in\Theta\}$ (e.g., linear model, neural net, probabilistic model).

In **global** (i.i.d.) models, $\theta_i\equiv\theta^\star$. In **context-adaptive** models, parameters vary with context:
$\theta_i=f(c_i)$ or $\theta_i\sim P(\theta\mid c_i)$.

For a new unit with context $c$, we write a unified empirical objective:
$$
\widehat{\theta}(c)\in\arg\min_{\theta\in\Theta}\;
\underbrace{\sum_{(i,j)\in S(c)} \ell\!\big(h_\theta(x_{ij}),y_{ij}\big)}_{\text{context-dependent support}}
\;+\;
\underbrace{\mathcal{R}(\theta;\,c)}_{\text{context-structured regularization}},
\tag{★}
$$
where $\ell$ is a proper loss (e.g., squared, logistic), $S(c)\subseteq\{1,\dots,n\}\times\mathbb{N}$ is a **support set** selected for context $c$, and $\mathcal{R}(\theta;c)$ encodes how parameters are allowed to vary with context (smoothness, sparsity, low-rank, hierarchy, etc.).

**How context enters.**  
- **Explicit parameterization:** a map $f:\mathcal{C}\to\Theta$ sets $\theta_i=f(c_i)$ (e.g., varying-coefficients, hierarchical Bayes, multi-task/meta-learning). Here $\mathcal{R}(\theta;c)$ typically regularizes $f$ (e.g., Lipschitz over $\mathcal{C}$, group lasso, low-rank).
- **Implicit parameterization:** context alters optimization or internal states without exposing $\theta$ directly (e.g., mixture-of-experts with gates $g(x,c)$; retrieval where $S(c)$ is built by a retriever $R(c)$; in-context learning where a prompt map $P(c)$ conditions a foundation model).

For convenience, we use a **context encoder** $\phi:\mathcal{C}\to\mathbb{R}^d$ and a similarity/kernel $K(c,c')$. A common instance of (★) is kernel-weighted risk:
$$
\sum_{i,j} w_{ij}(c)\,\ell\!\big(h_\theta(x_{ij}),y_{ij}\big)\;+\;\mathcal{R}(\theta),
\qquad
w_{ij}(c)\propto K\!\big(\phi(c),\phi(c_i)\big)\cdot \mathbf{1}\!\big[(i,j)\in S(c)\big].
$$

**Granularity.** We refer to adaptation granularity $g\in\{\text{group},\text{unit},\text{example}\}$ and to three design “knobs” we will revisit:  
1) **Information** via $S(c)$ or $P(c)$ (what context is exposed),  
2) **Inductive bias** via $\mathcal{R}(\theta;c)$ (how parameters may vary),  
3) **Compute** via warm-starts/caching/steps (how aggressively we solve (★) at test time).

**Standing assumptions (used as needed).**  
(i) *Exchangeability within context:* conditional on $(\theta_i,c_i)$, $(x_{ij},y_{ij})$ are i.i.d.  
(ii) *Regularity:* either $\theta=f(c)$ with $f$ in a regular class (e.g., Lipschitz/sparse/low-rank) or retrieval weights $w_{ij}(c)$ are bounded and locally normalized.  
(iii) *Identifiability/stability:* $\ell$ is convex in model outputs and $\mathcal{R}$ yields a unique or stable minimizer.  
(iv) *Resource tracking:* we track $|S(c)|$, optimization steps, and memory to compare **adaptation efficiency**.

With this notation in place, we now formalize the link between explicit and implicit context adaptation. This link has been contributed in pieces across recent work; here we unify those results.

### Theoretical Bridge

Recent theoretical work suggests that “explicit” context models (e.g., varying-coefficients, hierarchical/multitask) and “implicit” mechanisms (e.g., in-context learning via attention) often implement the same estimator class under squared loss, differing mainly in how they encode neighborhoods and regularization. Using the notation above, we make this precise.

**Proposition 1 (Explicit varying-coefficients and linear ICL coincide with kernel ridge on joint features in the linear squared-loss setting).**  
Assume squared loss and the regression model $y=\langle \theta(c),x\rangle+\varepsilon$ with $\mathbb{E}[\varepsilon]=0$. Let
(i) a context encoder $\phi:\mathcal{C}\to\mathbb{R}^{d_c}$,
(ii) joint features $\psi(x,c):=x\otimes \phi(c)\in\mathbb{R}^{d_x d_c}$,
(iii) a context-dependent support set $S(c)$ with nonnegative weights $w_{ij}(c)$.

- **(A) Explicit varying-coefficients.** Let $\theta(c)=B\,\phi(c)$ with $B\in\mathbb{R}^{d_x\times d_c}$ and ridge penalty $\lambda\lVert B\rVert_F^2$. The weighted ridge solution yields
  $$
  \widehat y(x,c)=k_{(x,c)}^\top \big(K+\lambda I\big)^{-1} y,\quad
  K_{ab}=\langle \psi_a,\psi_b\rangle=\langle x_a,x_b\rangle\cdot\langle \phi(c_a),\phi(c_b)\rangle,
  $$
  i.e., **kernel ridge regression (KRR)** on joint features.

- **(B) Implicit adaptation via linear ICL.** Let a single linear attention layer consume the weighted support set $S(c)$ with linear $q=Q\psi$, $k=K\psi$, $v=V\psi$ and a linear readout. With attention weights proportional to $w_{ij}(c)\cdot \langle q,k_{ij}\rangle$, the induced predictor equals KRR with kernel
  $$
  k\big((x,c),(x',c')\big)=\langle q(x,c),k(x',c')\rangle,
  $$
  i.e., a learned **dot-product kernel** on the same joint features. If attention parameters are trained in the linearized/NTK regime, learning equals kernel regression with the network’s NTK, which is again a dot-product kernel on linear transforms of $\psi$.

**Corollary 1 (Retrieval, gating, and weighting are kernel/measure choices).** Choosing $S(c)$ via a retriever $R(c)$, or gating in a mixture-of-experts, corresponds to changing the kernel and/or the empirical measure (weights $w_{ij}(c)$) used by KRR on $\psi$.

*Proof in Appendix A.*  

*Positioning and prior art.* Proposition 1 is expository: part (A) is standard ridge⇔kernel duality on joint features; part (B) follows from (i) fixed attention + trained linear head = ridge on fixed features and (ii) NTK linearization ⇒ kernel regression with the network’s NTK. 
Our contribution is the unified **context-aware** formulation: explicit design knobs via $S(c)$, $\mathcal{R}(\theta;c)$, and compute—and the mapping of retrieval/gating to kernel/measure choices. See transformer ICL as classical estimators [@arXiv:2212.07677; @arXiv:2208.01066; @arXiv:2212.10559] and NTK analyses [@arXiv:1806.07572; @arXiv:1912.02803].


### Scope of Review and Relation to Prior Work

In this review, we examine methods that use context to guide inference, either by specifying how parameters change with covariates or by learning to adapt behavior implicitly. 
We begin with classical models that impose explicit structure, such as varying-coefficient models and multi-task learning, and then turn to more flexible approaches like meta-learning and in-context learning with foundation models. 
Though these methods arise from different traditions, they share a common goal: to tailor inference to the local characteristics of each observation or task. 
Along the way, we highlight recurring themes: complex models often decompose into simpler, context-specific components; foundation models can both adapt to and generate context; and context-awareness challenges classical assumptions of homogeneity. 
These perspectives offer a unifying lens on recent advances and open new directions for building adaptive, interpretable, and personalized models.

#### Related Surveys and Reviews

Several surveys have examined specific aspects of context-adaptive inference, but they have largely remained confined to individual methodological traditions. 
Classical statistical surveys focus on varying-coefficient models and related structured regression methods. 
In machine learning, surveys on transfer and meta-learning emphasize task adaptation and shared representations, while recent work on foundation models explores the implicit adaptation capabilities of large pretrained models. 
Table 1 summarizes the scope and coverage of representative surveys.

| Survey | Topic Focus | Scope | Coverage of Adaptivity | Gap Relative to This Work |
|--------|-------------|-------|-------------------------|----------------------------|
| Statistical Methods with Varying Coefficient Models[@doi:10.4310/sii.2008.v1.n1.a15] | Varying-coefficient modeling | Classical statistical modeling, with parameters expressed as functions of covariates | Explicit adaptivity: parameters change smoothly with context via \(f(c)\) | Limited to explicit, parametric formulations; no connection to neural or emergent adaptation |
| A Survey of Deep Meta-Learning[@doi:10.48550/arXiv.2010.03522] | Meta-learning | Neural meta-learning methods for cross-task adaptation | Task-level adaptivity: models learn to generalize quickly across tasks | Focused on task switching; does not integrate explicit parameter modeling or implicit foundation model adaptation |
| LoRA: Low-Rank Adaptation of Large Language Models[@doi:10.48550/arXiv.2106.09685] | Parameter-efficient adaptation | Adaptation of large pretrained transformer models via low-rank updates while freezing base weights | Implicit adaptivity via parameter-efficient updates, enabling contextual adaptation without full fine-tuning | Strong in efficient adaptation mechanism, but narrow in scope; does not address explicit contextual structure or cross-domain generalization |
| Foundational Models Defining a New Era in Vision: A Survey and Outlook[@doi:10.48550/arXiv.2307.13721] | Vision-based foundation models | Architectures, multimodal integration, prompting, fusion in vision models | Implicit adaptivity in vision contexts, via prompt or fusion mechanisms across visual tasks | Domain-specific focus limits generalization; less discussion on theoretical adaptation across modalities |
| A Comprehensive Survey on Pretrained Foundation Models[@doi:10.48550/arXiv.2302.09419] | Pretrained foundation models | Coverage of models across modalities, training regimes, adaptation and fine-tuning strategies | Implicit adaptivity via representation transfer and generalization across tasks | Broad in scope but does not deeply analyze parameter-level adaptation or explicit–implicit alignment |

*Table 1: Representative surveys and key papers covering context-adaptive inference. Most works focus on a single methodological tradition and do not connect explicit and implicit approaches.*

While existing surveys have reviewed individual components of this landscape—such as varying-coefficient models, meta-learning, or foundation models—they have remained largely siloed.
This article provides the first comprehensive review that unifies explicit and implicit context-adaptive methods under a common framework.
By situating classical statistical models, modern machine learning methods, and foundation models along a shared spectrum of context-adaptive inference, we highlight common principles and distinctive challenges.
The next section outlines the conceptual foundations of context-adaptive inference, preparing the ground for detailed discussions of explicit and implicit modeling approaches in later sections.


## From Population Assumptions to Context-Adaptive Inference


Most statistical and machine learning models begin with a foundational assumption: that all samples are drawn independently and identically from a shared population distribution. This assumption simplifies estimation and enables generalization from limited data, but it collapses in the presence of meaningful heterogeneity.

In practice, data often reflect differences across individuals, environments, or conditions. These differences may stem from biological variation, temporal drift, site effects, or shifts in measurement context. Treating heterogeneous data as if it were homogeneous can obscure real effects, inflate variance, and lead to brittle predictions. As data grows more complex, the failure of this assumption not only limits accuracy but also obscures causal and contextual relationships underlying modern inference.

### Failure Modes of Population Models

Even when traditional models appear to fit aggregate data well, they may hide systematic failure modes.

**Mode Collapse**  
When one subpopulation is much larger than another, standard models are biased toward the dominant group, underrepresenting the minority group in both fit and predictions.

**Outlier Sensitivity**  
In the parameter-averaging regime, small but extreme groups can disproportionately distort the global model, especially in methods like ordinary least squares.

**Phantom Populations**  
When multiple subpopulations are equally represented, the global model may fit none of them well, instead converging to a solution that represents a non-existent average case.

![Failure Modes of Population Models. Illustrative schematics of common failure types when fitting a single global model to heterogeneous data. 
(A) **Mode Collapse**: the dominant group drives the fit, underrepresenting the minority. 
(B) **Outlier Sensitivity**: extreme points distort the global line, shifting predictions away from the majority. 
(C) **Phantom Populations**: the global fit represents no actual subgroup, but an artificial average. 
(D) **Hidden Confounding / Simpson’s Paradox**: aggregate trends reverse subgroup trends, obscuring true relationships.](images/population_failures.png){#fig:population-failures width="80%"}


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


A classical example of this challenge arises in causal inference. 
Following the Neyman–Rubin potential outcomes framework, we let $Y(1)$ and $Y(0)$ denote 
the outcomes that would be observed under treatment and control, respectively. 
The average treatment effect (ATE) is then $E[Y(1) - Y(0)]$, or more generally the conditional 
average treatment effect (CATE) given covariates. 
Standard approaches often condition only on $X$, while heterogeneous treatment effect (HTE) 
models incorporate additional context $C$ to capture systematic variation across subpopulations 
(Figure {@fig:hte-context}).

![Heterogeneous treatment effects. Left: average treatment effect (ATE) conditional on $X$, 
implicitly assuming homogeneity across contexts. Right: conditional average treatment effect (CATE) 
that allows treatment effects to vary systematically with additional context $C$.](images/hte.png){#fig:hte-context width="70%"}

These models highlight both the promise and the challenges of choosing and estimating $f(c)$.

### Classical Remedies: Grouped and Distance-Based Models

Before diving into flexible estimators of $f(c)$, we review early modeling strategies that attempt to break away from homogeneity.

#### Conditional and Clustered Models

One approach is to group observations into C contexts, either by manually defining conditions (e.g. male vs. female) or using unsupervised clustering. Each group is then assigned a distinct parameter vector:

$$
\{\widehat{\theta}_0, \ldots, \widehat{\theta}_C\} = \arg\max_{\theta_0, \ldots, \theta_C} \sum_{c \in \mathcal{C}} \ell(X_c; \theta_c),
$$
where $\ell(X; \theta)$ is the log-likelihood of $\theta$ on $X$ and $c$ specifies the covariate group that samples are assigned to. This reduces variance but limits granularity. It assumes that all members of a group share the same distribution and fails to capture variation within a group.

These early methods relax global homogeneity yet still rely on discrete partitions, motivating smoother and more flexible formulations explored in the next sections.

#### Distance-Regularized Estimation

A more flexible alternative assumes that observations with similar contexts should have similar parameters. This is encoded as a regularization penalty that discourages large differences in $\theta_i$ for nearby $c_i$:

$$
\{\widehat{\theta}_0, \ldots, \widehat{\theta}_N\} = \arg\max_{\theta_0, \ldots, \theta_N} \left( \sum_i \ell(x_i; \theta_i) - \sum_{i,j} \frac{\|\theta_i - \theta_j\|}{D(c_i, c_j)} \right),
$$

where $D(c_i, c_j)$ is a distance metric between contexts. This approach allows for smoother parameter variation but requires careful choice of $D$ and regularization strength $\lambda$ to balance bias and variance.  
The choice of distance metric D and regularization strength λ controls the bias–variance tradeoff.

### Parametric and Semi-parametric Varying-Coefficient Models

Varying-coefficient models (VCMs) provide one of the earliest formal frameworks for explicit adaptivity. Parametric VCMs assume that parameters vary linearly with covariates, a restrictive but interpretable assumption [@doi:10.1111/j.2517-6161.1993.tb01939.x]. The estimation can be written as
$$
\widehat{A} = \arg\max_A \sum_i \ell(x_i; A c_i).
$$
This formulation can be interpreted as a special case of distance-regularized estimation where the distance metric is Euclidean. Related developments in graphical models extend this idea to structured dependencies [@doi:10.1080/01621459.2021.2000866].

Semi-parametric VCMs relax the linearity assumption by requiring only that parameter variation be smooth. This is commonly encoded through kernel weighting, where the relevance of each sample is determined by its similarity in the covariate space [@doi:10.1214/aos/1017939139; @arxiv:2103.00315]. These models are more flexible but may fail when the true relationship between covariates and parameters is discontinuous.

### Contextualized Models

Contextualized models take a fully non-parametric approach, introduced in [@doi:10.48550/arXiv.1705.10301]. They assume that parameters are functions of context, $f(c)$, but do not restrict the form of $f$. Instead, $f$ is estimated directly, often with deep neural networks as function approximators:
$$
\widehat{f} = \arg \max_{f \in \mathcal{F}} \sum_i \ell(x_i; f(c_i)).
$$
This framework has been widely applied, from machine learning toolboxes [@doi:10.48550/arXiv.2310.11340; @doi:10.21105/joss.06469] to personalized genomics [@doi:10.1101/2023.12.01.569658; @doi:10.48550/arXiv.2111.01104], biomedical informatics [@doi:10.48550/arXiv.2310.07918; @doi:10.1016/j.jbi.2022.104086; @doi:10.1101/2020.06.25.20140053], and contextual feature selection [@doi:10.48550/arXiv.2312.14254]. These examples highlight how contextual signals can drive adaptation without assuming a fixed functional form.

### Partition and Latent-Structure Models

Partition models extend the contextualized framework by assuming that parameters can be divided into homogeneous groups, while leaving group boundaries to be inferred. This design is useful for capturing abrupt changes over covariates such as time. Estimation typically balances the likelihood with a penalty on parameter differences between adjacent samples, often expressed through a Total Variation (TV) penalty [@doi:10.1214/09-AOAS308]:
$$
\{\widehat{\theta}_0, \dots, \widehat{\theta}_N\} = \arg\max_{\theta_0, \dots, \theta_N} \left( \sum_i \ell(x_i; \theta_i) + \lambda \sum_{i = 2}^N \|\theta_i - \theta_{i-1}\| \right).
$$
By encouraging piecewise-constant structures, partition models get closer to personalized modeling, balancing fit and parsimony, moving closer to personalized inference, trading off flexibility for interpretability.

### Fine-tuned Models and Transfer Learning

Another practical strategy for handling heterogeneity is fine-tuning. A global population model is first estimated, and then a smaller set of parameters is updated for particular subpopulations. This idea underlies transfer learning, where large pre-trained models are adapted to new tasks with limited additional training [@doi:10.48550/arXiv.2206.02058]. Fine-tuning balances the bias–variance tradeoff by borrowing statistical strength from large datasets while preserving flexibility for local adaptation. This notion was already recognized in early VCM literature as a form of semi-parametric estimation [@doi:10.1214/aos/1017939139].

### Models for Explicit Subgroup Separation

Most adaptive methods encourage parameters for similar contexts to converge, but recent work explores the opposite: ensuring that models for distinct subgroups remain separated. This prevents minority subgroups from collapsing into majority patterns. Such “negative information sharing” is often implemented by learning representations that disentangle subgroup structure, bridging statistical partitioning with adversarial or contrastive learning objectives [@doi:10.48550/arXiv.1910.06939].

### A Spectrum of Context-Awareness

Context-aware models can be organized along a spectrum of assumptions about the relationship between context and parameters:

* **Global models**: $\theta_i = \theta$ for all $i$.
* **Grouped models**: $\theta_i = \theta_c$ for some finite set of groups.
* **Smooth models**: $\theta_i = f(c_i)$, with $f$ assumed to be continuous or low-complexity.
* **Latent models**: $\theta_i \sim P(\theta \mid c_i)$, with $f$ learned implicitly.

Each formulation encodes different beliefs about parameter variation. The next section formalizes these principles and examines general strategies for adaptivity in statistical modeling. For a discussion of how subpopulation shifts influence generalization, see [@arXiv:2106.04486].


### Independent and identically distributed samples

The initial research data mostly came from strictly designed experiments, such as agricultural field trials or psychological experiments. The characteristics of such data are small scale and simple structure. Researchers usually assume that each observation is independent of one another and identically distributed [@Fisher1949Design]. Under this setting, there is no dependency among the data, and researchers mainly focus on the overall average level or effect.

Linear models emerged as the fundamental approach to conduct statistical analysis for such data. One of the first methods in the development of linear models, the method of least squares, was first published in a work by Legendre [@Legendre1805Comets] and later independently developed and justified by Gauss [@Gauss1809Ambientium]. By reducing the squared deviations between predictions and results, this method offered a general framework for fitting a regression line via observed data. The estimator is expressed as follows:

$$
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{n} \bigl( y_i - x_i^{\top}\beta \bigr)^{2}
$$

which has a closed-form solution,

$$
\hat{\beta} = (X^{\top}X)^{-1}X^{\top}y
$$

This offered a systematic procedure for predicting unknown parameters from independently and identically distributed data, which builds the foundation of regression analysis.

The establishment of generalized linear models (GLMs) expanded the concepts of linear regression to non-Gaussian outcomes. Nelder and Wedderburn [@doi:10.2307/2344614] initiated the central concept of connecting the mean of the response variable to a linear predictor through a link function, which was later formalized into the now-standard compact notation [@doi:10.1007/978-1-4899-3242-6]:

$$
g(\mu_i) = \eta_i
$$

Before this unified general framework, one of the first instances was the use of the logistic function for binary data. Berkson advocated its application to biomedical dose–response studies, illustrating how the probabilities of success or failure in experimental settings could be captured by the link formulation [@doi:10.1080/01621459.1944.10500699]. Based on this, logistic regression was formalized as a regression model for binary sequences, establishing the logit link [@doi:10.1111/j.2517-6161.1958.tb00292.x]:

$$
\log \frac{p_i}{1-p_i} = x_i^\top \beta
$$

For count data, Poisson regression was introduced within the GLM framework, employing the log link [@doi:10.2307/2344614]:

$$
\log(\mu_i) = x_i^\top \beta
$$

These developments strengthened GLMs as a unifying framework for a variety of independently and identically distributed data types by extending linear modeling to categorical and count outcomes.

Alongside regression, analysis of variance (ANOVA) was another early milestone of statistical methodology. The development of ANOVA introduced the concept of splitting total variance into components related to within-group and between-group differences [@Fisher1925Research]. The central idea for ANOVA was the F ratio,

$$
F = \frac{MS_{\text{Between}}}{MS_{\text{Within}}}
$$

which provided a consistent framework for determining the significance of differences between group means and established the foundation of data analysis in modern experiments.

Together, these early statistical frameworks provided the foundation for analysis of independently and identically distributed data. However, as studies became more complex and observations were no longer truly independent, new methods were needed, leading to the development of hierarchical models.

### Hierarchical data

With the expansion of research fields, a hierarchical structure gradually emerges in the collected empirical data. For example, bull is nested in sire and herd [@doi:10.1080/00288233.1981.10420865]; the respondents are nested within the population of subjects [@doi:10.1037/h0040957]; and repeated measurements of the same object at different time points also make the data show longitudinal correlation. This type of data is more complex than independent and identically distributed samples. It not only has differences among groups but also needs to take into account the changes of individuals over time simultaneously. The observed values are no longer completely independent of each other, but there exists a distinct hierarchical effect.

Early work on hierarchical dependence started with the introduction of linear models that contain both fixed and random effects to estimate genetic parameters [@Henderson1950Genetic]. This approach laid the statistical basis for partitioning variability into systematic and random components, which became the basis for how intraclass correlation would later be understood and applied. A general form of the linear models can be written as:

$$
y_{ij} = \mu + u_j + \varepsilon_{ij}, \quad u_j \sim N(0, \sigma_u^2), \quad \varepsilon_{ij} \sim N(0, \sigma^2)
$$

Around the same period, it became clear that comparative rate estimation in clinical and epidemiological studies depends on the structure of sampled subgroups, with heterogeneity in source populations affecting inference in widespread use [@doi:10.1093/jnci/11.6.1269]. Building on this recognition of heterogeneity, applications expanded to repeated measurements and longitudinal structures; restricted maximum likelihood estimation was introduced to advance the framework and improve variance component inference in unbalanced settings [@doi:10.1093/biomet/58.3.545]. This methodological foundation enabled the formulation of linear mixed-effects models (LMMs) that combined fixed effects for population-level trends with random effects for subject- or group-specific deviations [@doi:10.2307/2529876]. In matrix notation, these models take the form

$$
y = X\beta + Zu + \varepsilon, \quad u \sim N(0, G), \quad \varepsilon \sim N(0, R)
$$

and later became widely used in the field of biostatistics and social sciences. To extend mixed-effects modeling beyond Gaussian responses, practical estimation procedures for generalized linear models with random effects were proposed, enabling the application of link functions to clustered binary, count, or categorical outcomes [@doi:10.1093/biomet/78.4.719]. Further methodological advances introduced penalized quasi-likelihood and approximate inference techniques that made practical application feasible in many fields, especially medical and biological fields [@doi:10.1080/01621459.1993.10594284]. These developments gave rise to Generalized Linear Mixed Models (GLMMs), formulated as

$$
g(\mu_i) = x_i^\top \beta + z_i^\top u
$$

combining link functions with both fixed ($\beta$) and random ($u$) effects. As mixed-effects approaches developed rapidly, the conceptual foundation of Bayesian hierarchical modeling—in which multilevel structures with prior distributions link parameters across groups—had already been articulated [@doi:10.1111/j.2517-6161.1972.tb00885.x]:

$$
y_{ij} \mid \theta_j \sim p(y_{ij} \mid \theta_j), \quad \theta_j \sim p(\theta_j \mid \phi), \quad \phi \sim p(\phi)
$$

Nevertheless, it wasn't until later that the Bayesian hierarchical model had practical influence. In the 1980s, the demonstration of the practical use of Gibbs sampling in image analysis paved the way for adoption in hierarchical Bayesian modeling [@doi:10.1109/TPAMI.1984.4767596]. Later, advances in Markov chain Monte Carlo methods provided the computational tools necessary to fit Bayesian hierarchical models and made Bayesian hierarchical modeling practical for applied researchers [@doi:10.1201/b16018].

Collectively, these developments shifted statistical modeling from independence assumptions to explicitly capturing correlation and hierarchical structure. As methods for nested data improved, new problems arose with functional, continuous, and high-dimensional observations. This led to the development of approaches for curves, trajectories, and large feature spaces, which in turn led to functional data analysis and high-dimensional inference.

### Functional types and high-dimensional data

As data collection advanced into the 1980s and 1990s, researchers began encountering observations that are entire curves or functions (e.g., time series, spectra, images) rather than fixed-dimensional vectors. Functional Data Analysis (FDA) [@doi:10.1002/0470013192.bsa239] treats each observation as a smooth function $x_i(t)$ defined over a continuum (time, wavelength, etc.). Typical FDA methods use basis expansions (e.g., Fourier or spline bases) to represent $x_i(t)$ and perform tasks such as functional principal component analysis. This emphasis on continuous predictors demanded flexible regression tools. Generalized Additive Models (GAMs) [@doi:10.1214/ss/1177013604] were developed to capture nonlinear effects while retaining interpretability. In a GAM, the response is modeled as

$$
y_i = \alpha + \sum_{j=1}^p f_j(x_{ij}) + \varepsilon_i
$$

where each $f_j$ is an estimated smooth function (often implemented with splines or kernels). GAMs then generalize linear models by allowing the data to determine the shape of each predictor’s effect. For example, a GAM can automatically reveal a U-shaped or threshold effect in a covariate, which a purely linear model would miss.

As data dimensionality and complexity continued to grow (for instance with genomic or imaging data), there was a shift toward automated feature learning. Representation learning [@doi:10.1109/TPAMI.2013.50] generalizes classical dimension reduction (like PCA) to nonlinear, data-driven embeddings. Neural autoencoders are a prototypical example: one trains an encoder network $h_\phi(x)$ and a decoder $g_\theta(z)$ by minimizing reconstruction error,


$$
(\theta^{\\ast},\phi^{\\ast}) = \arg\min_{\theta,\phi}\sum_{i=1}^n |x_i - g_\theta(h_\phi(x_i))|^2
$$

Here $z_i = h_\phi(x_i)$ is a low-dimensional code capturing the essential structure of $x_i$. With linear networks this recovers PCA, but in general the learned code can represent highly nonlinear features. Modern representation learning methods (e.g., deep autoencoders, variational autoencoders, deep embedding networks) therefore enable models to adapt their feature extraction to complex, high-dimensional data in a context-dependent way [@doi:10.1109/TPAMI.2013.50]. By learning bases and transformations from the data itself, these methods built on the ideas of FDA and GAMs to handle the evolving scale and richness of scientific data.

### Heterogeneous tasks and sparse data

With the proliferation of diverse tasks and domains, statistical learning shifted toward methods that transfer information across problems. Multi-task learning (MTL) [@doi:10.1023/A:1007379606734] arose in response: it jointly models related tasks to improve performance, especially when each task has limited data. Concretely, if we have $T$ tasks with data $(X^t,Y^t)$ for $t=1,\dots,T$, one can learn task-specific models $w^t$ by minimizing a coupled objective, for example

$$
\min_{W}\sum_{t=1}^T \sum_{i=1}^{n_t} \ell(y_i^t, f(x_i^t;w^t)) + \lambda\,\Omega(W)
$$

where $W=[w^1,\dots,w^T]$ collects all task parameters and $\Omega(W)$ is a regularizer that enforces shared structure (e.g., penalizing deviations from a common parameter). This formulation lets tasks share a representation or bias: information useful for one task can aid others. Transfer learning extends this idea between domains wherein a model learned on a large source dataset is adapted to a target domain with little labeled data [@doi:10.1109/TKDE.2009.191]. For instance, a convolutional neural network pretrained on ImageNet can be fine-tuned on a small medical imaging dataset, effectively reusing learned features to improve accuracy with scarce data.

Another fundamental challenge is distribution shift between training and deployment. Covariate shift occurs when the input distribution $p(x)$ changes but the conditional $p(y|x)$ remains the same. In this case one can correct the loss by importance weighting. Denote $p_{\mathrm{train}}(x)$ and $p_{\mathrm{test}}(x)$ the feature distributions; then

$$
\mathbb{E}_{x\sim p_{\mathrm{test}}}[\ell(f(x),y)]
= \mathbb{E}_{x\sim p_{\mathrm{train}}}\Bigl[\tfrac{p_{\mathrm{test}}(x)}{p_{\mathrm{train}}(x)}\,\ell(f(x),y)\Bigr], \quad
w(x) = \frac{p_{\mathrm{test}}(x)}{p_{\mathrm{train}}(x)}
$$

More generally, domain adaptation methods address cases where both $p(x)$ and $p(y|x)$ differ across source and target domains. Techniques such as kernel mean matching, adversarial domain-invariant representations, or graph-based alignment were developed to tackle these shifts. These methods emerged as data collection became distributed across different environments (e.g., new sensors, populations, or changing conditions), requiring models that adapt to new contexts beyond the original training distribution.

Finally, the combination of many tasks and very few examples per task has given rise to few-shot learning. In few-shot learning [@doi:10.1145/3386252], the goal is to generalize to new classes or tasks from only a handful of labeled examples. Modern approaches typically leverage prior experience across tasks or classes. Metric-based methods, such as Matching Networks and Prototypical Networks [@doi:10.48550/arXiv.1606.04080; @doi:10.48550/arXiv.1703.05175], learn an embedding $\phi(x)$ so that samples from the same class cluster together. A novel class can then be represented by the mean of its few support examples in this latent space. Alternatively, meta-learning algorithms train on many simulated few-shot tasks so the model learns to adapt quickly. These few-shot and meta-learning frameworks explicitly confront data scarcity by transferring inductive biases across tasks, enabling effective learning in regimes far beyond the traditional large-sample assumptions.

### Online and interactive data

In the Internet era, the way data is collected has undergone significant changes. User behavior data, sensor data, and platform experimental data exhibit the characteristics of streaming and interactivity [@doi:10.1145/1772690.1772758]. The generation of dynamic data poses new challenges to model construction and problem-solving. Data is not collected all at once but is generated in real time and is often related to the feedback loop of the system [@doi:10.1115/1.3662552]. This type of data is more dynamic and complex compared to traditional experimental or measurement data, involving both time dependence and continuous influence from the environment and interaction.

A natural starting point to address this challenge is the concept of online learning [@doi:10.48550/arXiv.1802.02871]. A foundational formulation is the Online Convex Optimization (OCO) framework [@doi:10.1561/2200000018]. At each round $t = 1, \dots, T$, the learner selects a decision $x_t \in F$, where $F \subset \mathbb{R}^n$ is a convex decision set. The environment then reveals a convex loss function $c_t : F \to \mathbb{R}$, and the learner incurs loss $c_t(x_t)$. The central performance measure is the regret, defined as:

$$
R(T) = \sum_{t=1}^{T} c_t(x_t) - \min_{x \in F} \sum_{t=1}^{T} c_t(x)
$$

which compares the learner’s cumulative loss with that of the best fixed decision in hindsight. A simple and influential algorithm in this framework is Online Gradient Descent (OGD). Given a subgradient $g_t \in \partial c_t(x_t)$, OGD performs a gradient step followed by projection back onto the feasible set:

$$
x_{t+1} = \Pi_F(x_t - \eta_t g_t)
$$

where $\eta_t$ is a step size and $\Pi_F$ denotes projection onto $F$. With appropriately chosen step sizes, OGD achieves the classic bound $R(T) = O(\sqrt{T})$, implying vanishing average regret. This framework provides the mathematical foundation for more complex online methods, such as bandits and reinforcement learning.

While OCO provides guarantees with respect to a fixed comparator, real-world data streams rarely remain stationary. The statistical relationship between inputs $x$ and outputs $y$ can change over time. This phenomenon is known as concept drift [@doi:10.1145/2523813]. To cope with drift, online adaptive learning methods augment the basic OCO paradigm by explicitly adapting to distributional changes. A representative technique for drift detection is the Drift Detection Method (DDM) [@doi:10.1007/978-3-540-28645-5_29]. It monitors the online error rate $\hat{p}\_i$ after $i$ samples together with its standard deviation $s\_i$. The method keeps track of the minimum values observed $(p\_{\min}, s\_{\min})$, and raises an alarm when

$$
\hat{p}_i + s_i \geq p_{\min} + \alpha \cdot s_{\min}
$$

for a predefined threshold $\alpha$. This statistical test signals when the current error distribution significantly deviates from the past minimum, indicating concept drift. The old model is then discarded, and a new model is trained using the data accumulated since the warning level (i.e., from $k_w$ to $k_d$). After reinitialization, $(p_{\min}, s_{\min})$ are reset, and the updated model continues processing the incoming stream. Beyond DDM, the Early Drift Detection Method (EDDM) [@Baena2006EarlyDrift] monitors the distribution of distances between errors, making it more sensitive to gradual drifts. Ensemble-based methods such as Bagging-ADWIN and Boosting-ADWIN [@doi:10.1145/1557019.1557041], which couple classical ensemble learning with adaptive sliding windows, have also demonstrated strong performance on evolving data streams.

Though adaptive learning addresses the challenge of non-stationarity, it still operates under a full-information feedback model: after each round, the entire loss function $c_t(\cdot)$ is revealed. In many interactive systems, however, such complete feedback is unavailable. This setting motivates the study of partial-information online learning, captured by the classic Multi-Armed Bandit (MAB) framework [@doi:10.1090/S0002-9904-1952-09620-8]. Within the MAB, the agent repeatedly chooses among a finite set of actions (“arms”) and receives the reward of that arm. The central challenge is the exploration–exploitation trade-off. Classical strategies include $\epsilon$-greedy [@Sutton1998RLIntroduction], which with probability $1-\epsilon$ selects the current best arm and with probability $\epsilon$ explores. Upper Confidence Bound (UCB) algorithms [@doi:10.1023/A:1013689704352] emphasize the “Optimism in the Face of Uncertainty,” where at each round $t$ with arm $a$:

$$
\mathrm{UCB}_a(t) = \hat{\mu}_a + \sqrt{\frac{2\ln t}{n_a}}
$$

The arm $\arg\max_a \mathrm{UCB}_a(t)$ is chosen at each round, achieving sublinear regret. Their key limitation—ignoring contextual side information—leads naturally to Contextual Bandits. Formally, at each round $t$, the agent now observes a context $x_t$ and the goal is to learn a policy $\pi : X \to A$ that maximizes expected cumulative reward. A representative algorithm is LinUCB, which assumes a linear relationship between context and reward; at round $t$, LinUCB selects

$$
a_t = \arg \max_{a \in \mathcal{A}} \left( x_t^\top \hat{\theta}_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right)
$$

where $\hat{\theta}_a$ is the estimated weight and $A_a$ is the design matrix. Contextual bandits improve by personalizing decisions to the observed environment. They have been widely applied in news recommendation and adaptive experimentation. However, CB assumes actions do not affect future contexts, which motivates the richer framework of reinforcement learning.

RL formalizes sequential decision-making under uncertainty through the lens of Markov Decision Processes (MDPs) [@Bellman1957Markovian]. An MDP is defined by a state space $S$, an action space $A$, transition dynamics $P(s' \mid s, a)$, a reward function $r(s, a)$, and a discount factor $\gamma \in [0, 1)$. At each step $t$, the agent observes a state $s_t$, selects an action $a_t \sim \pi(\cdot \mid s_t)$, receives a reward $r_t$, and transitions to a new state $s_{t+1}$. The goal is to learn a policy $\pi$ that maximizes the expected cumulative discounted reward:

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

Classical solution approaches include value-based methods (e.g., Q-learning) [@Watkins1989Learning], which estimate action-value functions and act greedily with respect to them, and policy-based methods (e.g., policy gradient, actor–critic) [@doi:10.1023/a:1022672621406] [@doi:10.1109/tsmc.1983.6313077], which directly optimize the policy parameters via stochastic gradient ascent. These methods establish theoretical guarantees and have enabled applications ranging from game-playing (e.g., Go, Atari) [@doi:10.1038/nature14236] to robotics.

The trajectory from online convex optimization to reinforcement learning highlights how the statistical study of streaming, interactive data has evolved into increasingly expressive frameworks for context-adaptive inference. As data collection becomes more interactive and non-stationary, the ability to learn from context and feedback loops is central to adaptive intelligence. This theme naturally connects to the next section on multimodal data, where adaptivity must also span across modalities.

### Multimodal data

With the advancement of the digitalization process, research and application have begun to simultaneously involve various types of data such as images [@doi:10.1088/2040-8978/17/7/073001], audio [@doi:10.1121/1.1912679], and text [@doi:10.1145/361219.361220]. These data are not only high-dimensional but also show clear structural and representational differences. Text consists of discrete symbolic sequences that contain semantic and grammatical structures; images are spatially organized pixel matrices that reflect local spatial correlations and global patterns; and audio is a continuous waveform signal that captures dynamic temporal characteristics. Compared with earlier single-type numerical or functional data, multimodal data are more heterogeneous and complex. New challenges asked researchers to explore how to establish semantic correspondences across modalities for cross-modal alignment, build joint models through shared latent spaces, and dynamically adjust inter-modal dependencies according to different tasks or contexts.

From a statistical perspective, representation learning [@doi:10.1109/TPAMI.2013.50] can be regarded as a generalization of traditional linear dimension reduction techniques such as Principal Component Analysis (PCA) and Factor Analysis (FA). While PCA and FA aim to identify low-dimensional subspaces that capture maximum variance or shared covariance structure through linear projections, representation learning extends this idea to nonlinear mappings that can model highly complex and structured data. Representation learning addresses this challenge by automatically learning low-dimensional embeddings that map complex observations into a shared latent space. The basic idea is to learn a nonlinear mapping

$$
f_{\theta}: \mathcal{X} \rightarrow \mathcal{Z}
$$

where $\mathcal{X}$ represents the original multimodal input space (e.g., images, text, or audio), and $\mathcal{Z}$ denotes the shared latent space. The context-adaptive nature of representation learning lies in its ability to adjust feature extraction pathways according to the input modality and to align heterogeneous data within a unified semantic space.

To infer hidden structures, traditional latent variable models in statistics such as Factor Analysis and Gaussian Mixture Models (GMM) require estimating the posterior distribution $p(z|x)$. However, when the generative process $p_\theta(x|z)$ is highly nonlinear, the posterior becomes intractable. Variational inference approximates the true posterior by introducing a tractable distribution $q(z|x)$ and minimizing their divergence, typically via the Kullback–Leibler (KL) divergence.

Amortized inference introduces a major innovation by parameterizing the approximate posterior as a learnable function $q_\phi(z|x)$ shared across all samples. This amortizes the cost of inference, enabling efficient posterior estimation via a neural network encoder. Variational Autoencoders (VAEs) [@arXiv:1312.6114] can be regarded as nonlinear extensions of classical latent variable models. VAEs jointly train the generative model $p_\theta(x|z)$ and the inference model $q_\phi(z|x)$ by maximizing the Evidence Lower Bound (ELBO):

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}[q_\phi(z|x) \,||\, p(z)]
$$

The model can dynamically adjust posterior estimation $q_\phi(z|x)$ according to input characteristics or modality, achieving efficient and scalable probabilistic reasoning.

In many real-world scenarios, the amount of labeled data for each task may be limited. Meta-learning, or learning to learn, provides a framework for acquiring inductive knowledge that enables rapid adaptation to new tasks with few examples. The central idea is to learn a meta-model that captures transferable structures across tasks and can quickly adjust its parameters when facing a new context. It involves a two-level optimization process:

$$
\min_{\theta} \sum_{T_i \sim p(T)} \mathcal{L}_{T_i}\big(U(\theta, T_i)\big)
$$

where $\theta$ denotes the meta-parameters shared across tasks, $T_i$ represents individual tasks sampled from a task distribution $p(T)$, and $U(\theta, T_i)$ indicates the task-specific adaptation process, such as gradient updates.

Two main paradigms have been developed. Gradient-based meta-learning, exemplified by Model-Agnostic Meta-Learning (MAML) [@doi:10.48550/arXiv.1703.03400], optimizes an initialization of $\theta$ that allows fast convergence on new tasks with a few gradient steps. In contrast, metric-based meta-learning [@doi:10.48550/arXiv.1703.05175] focuses on learning a task-invariant representation space in which samples from the same class or semantic context remain close in distance metrics.

Meta-learning can be viewed as a hierarchical model where the outer loop learns the hyperprior over tasks and the inner loop performs task-specific inference. The model internalizes the variability across tasks into shared meta-parameters and dynamically adjusts its learning strategy when exposed to new contextual distributions. Such an approach bridges the gap between multi-task learning and context-aware adaptation, providing a scalable solution for heterogeneous and data-sparse environments.

### Large-scale pre-trained data

With the rapid expansion of digital ecosystems and the emergence of web-scale data collection, research has entered a new stage characterized by massive, heterogeneous, and weakly supervised datasets. In recent years, a vast amount of cross-domain data has been centrally collected—such as large-scale text corpora [@doi:10.48550/arXiv.1810.04805] and multimodal alignment data [@doi:10.48550/arXiv.2103.00020] that combine text, images, audio, and sound—representing an unprecedented scale and diversity of information sources. Unlike earlier multimodal datasets that were carefully curated for specific experimental designs, these pretraining datasets are gathered under open, non-controlled conditions and often lack explicit task labels or unified structures.

The central modeling challenge has shifted from fitting a single data distribution to extracting transferable and generalizable patterns that remain stable across heterogeneous contexts. This transition has motivated the development of foundation models and related adaptive paradigms, which aim to capture universal representations, perform context-dependent inference, and achieve robust generalization across domains.

Foundation models [@doi:10.48550/arXiv.2108.07258] represent a fundamental paradigm shift in machine learning toward building universal systems trained on massive and heterogeneous data sources. Unlike earlier models that were designed for specific tasks or modalities, foundation models learn general-purpose representations through large-scale pretraining, which can be adapted to downstream tasks with minimal fine-tuning. Given a massive dataset $\mathcal{D} = \lbrace x_i \rbrace_{i=1}^N$ sampled from diverse domains, the objective is to learn a parameterized function $f_\theta$ that captures broadly transferable patterns:

$$
\theta^* = \arg\min_\theta \; \mathbb{E}_{x \sim \mathcal{D}} \, \ell(f_\theta(x))
$$

where $\ell$ represents a pretraining objective such as next-token prediction or contrastive alignment.

The context-adaptive nature of foundation models arises from their ability to integrate knowledge from large and diverse data distributions, enabling emergent behaviors such as cross-modal transfer, domain generalization, and zero-shot reasoning. They thus redefine the notion of generalization—from fitting within a context to adapting across contexts—by embedding statistical invariances into large-scale learned representations.

In-context learning (ICL) [@doi:10.48550/arXiv.2301.00234] describes the ability of large language or multimodal models to adapt to new tasks through contextual examples provided at inference time, without updating model parameters. Given a prompt sequence $\lbrace(x_i, y_i)\rbrace_{i=1}^k$ followed by a query $x_{k+1}$, the model generates a prediction $\hat{y}_{k+1}$ by conditioning on the in-context information:

$$
\hat{y}_{k+1} = f_\theta(x_{k+1} \mid x_1, y_1, \dots, x_k, y_k)
$$

This phenomenon suggests that pretrained models can perform implicit meta-learning within their internal representations, dynamically adjusting inference behavior according to contextual input. From a statistical perspective, ICL transforms the adaptation process from an external optimization (parameter update) to an internal inference mechanism conditioned on observed data, thus embodying a new form of context-adaptive reasoning.



## Principles of Context-Adaptive Inference

What makes a model adaptive? When is it good for a model to be adaptive? While the appeal of adaptivity lies in flexibility and personalized inference, not all adaptivity is beneficial. This section formalizes the core principles that underlie adaptive modeling and situates them within both classical statistics and recent advances in machine learning.

Adaptivity is best understood as a structured set of design principles rather than a single mechanism. Each principle described below highlights a different axis along which models can incorporate or restrict adaptation. Flexibility captures the representational capacity needed for adaptation, while signals of heterogeneity determine when adaptation is justified. Modularity helps organize adaptation into interpretable and transferable units, and selectivity guards against overfitting by controlling when adaptation is triggered. Data efficiency limits how finely we can adapt in practice, and tradeoffs remind us that adaptation is never free of cost. Together, these principles delineate both the potential and the pitfalls of adaptive systems.

We organize this section around six core principles: flexibility, heterogeneity signals, modularity, selectivity, data efficiency, and tradeoffs. Afterward, we discuss failure modes and conclude with a synthesis that connects these ideas to practical implications.

### 1. Adaptivity requires flexibility
The first principle concerns model capacity. A model must be able to represent multiple behaviors if it is to adapt. Without sufficient representational richness, adaptation becomes superficial, amounting only to noise-fitting rather than meaningful personalization. Flexibility provides the foundation for models to express diverse responses across individuals, groups, or environments, rather than enforcing a single global rule.

Flexibility may arise from different modeling strategies. In classical statistics, regression models with interaction effects explicitly capture how predictors influence outcomes differently across contexts, while hierarchical and multilevel models let effects vary systematically across groups. Varying-coefficient models extend this further by allowing regression coefficients to evolve smoothly with contextual covariates [@doi:10.1111/j.2517-6161.1993.tb01939.x]. In machine learning, meta-learning and mixture-of-experts architectures [@doi:10.1162/neco.1991.3.1.79] offer dynamic allocation of capacity, training models to specialize on tasks or inputs as needed. Together, these approaches illustrate the common principle that without flexibility, adaptation has no meaningful space in which to operate.

### 2. Adaptivity requires a signal of heterogeneity
Flexibility alone is not enough; a model also requires observable signals that indicate how and why adaptation should occur. Without such signals, adaptive systems risk reacting to random fluctuations rather than capturing meaningful structure. In statistics, varying-coefficient regressions illustrate this idea by allowing parameters to change smoothly with observed covariates [@doi:10.1111/j.2517-6161.1993.tb01939.x], while hierarchical models assume systematic group differences that provide a natural signal for adaptive pooling.

In machine learning, contextual bandits adapt decisions to side information that characterizes the current environment, while benchmarks like WILDS highlight that real-world datasets often contain distributional shifts and subgroup heterogeneity [@doi:10.48550/arXiv.2012.07421]. Recent work extends this further, modeling time-varying changes in continuous temporal domain generalization [@doi:10.48550/arXiv.2405.16075] or leveraging diversity across experts to separate stable from unstable patterns [@doi:10.48550/arXiv.2410.17020]. Across applications, from medicine to online platforms, heterogeneity signals provide the essential cues that justify adaptation.

### 3. Modularity improves adaptivity
Organizing adaptation into modular units improves interpretability and robustness. Instead of spreading changes across an entire system, modularity restricts variation to well-defined subcomponents that can be recombined, reused, or replaced. This structure provides three advantages: targeted adaptation, transferability across tasks, and disentanglement of variation sources.

A canonical example is the mixture-of-experts framework, where a gating network routes inputs to specialized experts trained for different data regimes [@doi:10.1162/neco.1991.3.1.79]. By decomposing capacity in this way, models not only gain efficiency but also clarify which components are responsible for specific adaptive behaviors. Recent advances extend this principle in modern architectures: modular domain experts [@doi:10.48550/arXiv.2410.10181], adapter libraries for large language models [@doi:10.48550/arXiv.2405.11157], and mixtures of LoRA experts [@doi:10.48550/arXiv.2404.13628]. In applications ranging from language processing to computer vision, modularity has become a cornerstone of scalable adaptivity.

### 4. Adaptivity implies selectivity
Adaptation must not occur indiscriminately. Overreacting to noise leads to overfitting, defeating the purpose of adaptation. Selectivity provides the discipline that ensures adaptive mechanisms respond only when supported by reliable evidence.

Classical statistics formalized this principle through methods such as Lepski’s rule for bandwidth selection, which balances bias and variance in nonparametric estimation [@doi:10.1214/aos/1030741083]. Aggregation methods such as the weighted majority algorithm show how selective weighting of multiple models can improve robustness [@doi:10.1006/inco.1994.1009]. In modern machine learning, Bayesian rules can activate test-time updates only when uncertainty is manageable [@doi:10.48550/arXiv.2410.03306], while confidence-based strategies prevent unstable adjustments by holding back adaptation under weak signals [@doi:10.48550/arXiv.2310.04941]. Sparse expert models apply the same principle architecturally, activating only a few experts for easy inputs but engaging more capacity for difficult cases [@doi:10.48550/arXiv.2403.07652]. These safeguards demonstrate that good adaptation is selective adaptation.

### 5. Adaptivity is bounded by data efficiency
Even with flexibility, heterogeneity, modularity, and selectivity in place, the scope of adaptation is fundamentally constrained by the availability of data. Fine-grained adaptation requires sufficient samples to estimate context-specific effects reliably. When data are scarce, adaptive systems risk inflating variance, capturing noise, or overfitting to idiosyncratic patterns. This limitation transcends individual methods and reflects a general statistical truth.

Meta-learning research illustrates this tension, as few-shot frameworks show both the promise of cross-task generalization and the sharp degradation that occurs when task diversity or sample size is insufficient [@doi:10.48550/arXiv.1810.02334]. Bayesian analyses of scaling laws for in-context learning formalize how the reliability of adaptation grows with data [@doi:10.48550/arXiv.2410.16531]. To mitigate these limits, modular reuse strategies have been developed, including adapter libraries [@doi:10.48550/arXiv.2405.11157] and modular domain experts. Practical applications, from medicine to recommendation systems, highlight the same lesson: adaptation cannot outpace the data that supports it.

### 6. Adaptivity is not a free lunch
Adaptivity brings benefits yet inevitably incurs costs. It can reduce bias and improve personalization, but at the expense of variance, computational resources, and stability. A model that adapts too readily may become fragile, inconsistent across runs, or difficult to interpret.

In statistical terms, this tension is captured by the classic bias and variance tradeoff [@doi:10.1109/72.788640]: increasing flexibility reduces systematic error but simultaneously increases estimation variance, especially in small-sample settings. Adaptive methods expand flexibility, which means they must also contend with this cost unless constrained by strong regularization or selectivity. In machine learning practice, these tradeoffs surface in multiple ways. Sparse expert models illustrate them clearly: while they scale efficiently, routing instability can cause experts to collapse or remain underused, undermining reliability [@doi:10.48550/arXiv.2406.18219]. Test-time adaptation can boost performance under distribution shift but may destabilize previously well-calibrated predictions. These examples show that adaptation is powerful but never free.

### When Adaptivity Fails: Common Failure Modes
The six principles describe when adaptation should succeed, but in practice, failures remain common. Understanding these failure modes is crucial for designing safeguards, as they reveal the vulnerabilities of adaptive methods when principles are ignored or misapplied. Failure does not imply that models lack adaptivity, but that adaptation proceeds in unstable or unjustified ways.

**Spurious adaptation.** Models sometimes adapt to unstable or confounded features that correlate with outcomes only transiently. This phenomenon is closely related to shortcut learning in deep networks, where spurious correlations masquerade as useful signals [@doi:10.48550/arXiv.2004.07780; @doi:10.48550/arXiv.2012.07421]. Such adaptation may appear effective during training but fails catastrophically under distribution shift. The lesson here is that models must rely on stable signals of heterogeneity, not superficial correlations.

**Overfitting in low-data contexts.** Fine-grained adaptation requires sufficient signal. When the available data are limited, adaptive models tend to inflate variance and personalize to noise rather than meaningful structure. Meta-learning research illustrates this tension: although few-shot methods aim to generalize with minimal samples, they often degrade sharply when task diversity is low or heterogeneity is weak [@doi:10.48550/arXiv.1810.02334]. This failure mode underscores the principle that data efficiency sets unavoidable limits on adaptivity.

**Modularity mis-specification.** Although modularity can improve interpretability and transfer, poorly designed modules or unstable routing mechanisms can create new sources of error. Group-shift robustness studies reveal that when partitions are misaligned with true structure, adaptive pooling can worsen disparities across groups [@doi:10.48550/arXiv.1911.08731]. Similarly, analyses of mixture-of-experts models show that mis-specified routing can cause experts to collapse or remain underutilized [@doi:10.48550/arXiv.2406.18219]. These cases highlight that modularity is beneficial only when aligned with meaningful heterogeneity.

**Feedback loops.** Adaptive models can also alter the very distributions they rely on, especially in high-stakes applications such as recommendation, hiring, or credit scoring. This creates feedback loops where bias is reinforced rather than corrected. For example, an adaptive recommender system that over-personalizes may restrict exposure to diverse content, reshaping user behavior in ways that amplify initial bias. The selective labels problem in algorithmic evaluation illustrates how unobserved counterfactuals complicate learning from adaptively collected data [@doi:10.1145/3097983.3098066]. These examples show that adaptation must be evaluated with attention to long-term interactions, not only short-term accuracy.

Taken together, these failure modes illustrate that adaptivity is double-edged: the same mechanisms that enable personalization and robustness can also entrench bias, waste data efficiency, or destabilize models if not carefully designed and monitored.

![Failure Modes of Context-Adaptive Models.
(A) **Mode Collapse**: adaptive fits diverge from the true stable relationship.
(B) **Overfitting in Low-Data Contexts**: adaptation follows noise rather than signal.
(C) **Modularity Mis-Specification**: incorrect partitions obscure the true structure.
(D) **Feedback Loops**: adaptive decisions reshape the very data they rely on.](images/adaptive_failures.png){#fig:adaptive-failures width="80%"}

Having examined when and why adaptivity fails, we now synthesize these insights into a set of guiding principles for practical model design.

### Synthesis and Implications
The principles and failure modes together provide a coherent framework for context-adaptive inference. Flexibility and heterogeneity define the capacity and justification for adaptation, ensuring that models have room to vary and meaningful signals to guide that variation. Modularity and selectivity organize adaptation into structured, interpretable, and disciplined forms, while data efficiency and tradeoffs impose the practical limits that prevent overreach. Failure modes remind us that these principles are not optional: neglecting them can lead to spurious adaptation, instability, or entrenched bias.

For practitioners, these insights translate into a design recipe. Begin by ensuring sufficient flexibility, but constrain it through modular structures that make adaptation interpretable and transferable. Seek out reliable signals of heterogeneity that justify adaptation, and incorporate explicit mechanisms of selectivity to guard against noise. Respect the limits imposed by data efficiency, recognizing that fine-grained personalization requires sufficient statistical support. Always weigh the tradeoffs explicitly, balancing personalization against stability, efficiency against interpretability, and short-term gains against long-term robustness. Evaluation criteria should extend beyond predictive accuracy to include calibration, fairness across subgroups, stability under distributional shift, and resilience to feedback loops.

By connecting classical statistical models with modern adaptive architectures, this framework provides both a conceptual map and practical guidance. It highlights that context-adaptive inference is not a single technique but a set of principles that shape how adaptivity should be designed and deployed. When applied responsibly, these principles enable models that are flexible yet disciplined, personalized yet robust, and efficient yet interpretable. Building on these conceptual principles, we next examine how context-adaptive inference can be made computationally and statistically efficient.




### Context-Aware Efficiency Principles and Design

The efficiency of context-adaptive methods hinges on several key design principles that balance computational tractability with statistical accuracy. These principles guide the development of methods that can scale to large datasets while maintaining interpretability and robustness.

Context-aware efficiency often relies on sparsity assumptions that limit the number of context-dependent parameters. This can be achieved through group sparsity, which encourages entire groups of context-dependent parameters to be zero simultaneously [@doi:10.1111/j.1467-9868.2005.00532.x], hierarchical regularization that applies different regularization strengths to different levels of context specificity [@doi:10.1111/j.2517-6161.1996.tb02080.x; @doi:10.1017/CBO9780511790942], and adaptive thresholding that dynamically adjusts sparsity levels based on context complexity.

Efficient context-adaptive inference can be achieved through computational strategies that allocate resources based on context. Early stopping terminates optimization early for contexts where convergence is rapid [@doi:10.48550/arXiv.1606.04838], while context-dependent sampling uses different sampling strategies for different contexts [@doi:10.48550/arXiv.1809.09582]. Caching and warm-starting leverage solutions from similar contexts to accelerate optimization, particularly effective when contexts exhibit smooth variation [@doi:10.1561/2200000016].

The design of context-aware methods often involves balancing computational efficiency with interpretability. Linear context functions are more interpretable but may require more parameters, while explicit context encoding improves interpretability but may increase computational cost. Local context modeling provides better interpretability but may be less efficient for large-scale applications. These trade-offs must be carefully considered based on the specific requirements of the application domain, as demonstrated in recent work on adaptive optimization methods [@doi:10.48550/arXiv.1412.6980].

### Adaptivity is bounded by data efficiency

Recent work underscores a practical limit: stronger adaptivity demands more informative data per context. When contexts are fine-grained or rapidly shifting, the effective sample size within each context shrinks, and models risk overfitting local noise rather than learning stable, transferable structure. Empirically, few-shot behaviors in foundation models improve with scale yet remain sensitive to prompt composition and example distribution, indicating that data efficiency constraints persist even when capacity is abundant [@doi:10.48550/arXiv.2005.14165; @doi:10.48550/arXiv.2206.07682; @doi:10.48550/arXiv.2202.12837]. Complementary scaling studies quantify how performance depends on data, model size, and compute, implying that adaptive behaviors are ultimately limited by sample budgets per context and compute allocation [@doi:10.48550/arXiv.2001.08361; @doi:10.48550/arXiv.2203.15556; @doi:10.48550/arXiv.2410.16531]. In classical and modern pipelines alike, improving data efficiency hinges on pooling information across related contexts (via smoothness, structural coupling, or amortized inference) while enforcing capacity control and early stopping to avoid brittle, context-specific artifacts [@doi:10.48550/arXiv.1606.04838]. These considerations motivate interpretation methods that report not only attributions but also context-conditional uncertainty and stability, clarifying when adaptive behavior is supported by evidence versus when it reflects data scarcity.

#### Formalization: data-efficiency constraints on adaptivity

Let contexts take values in a measurable space $\mathcal{C}$, and suppose the per-context parameter is $\theta(c) \in \Theta$.  
For observation $(x,y,c)$, consider a conditional model $p_\theta(y \mid x,c)$ with loss $\ell(\theta; x,y,c)$.  
For a context neighborhood $\mathcal{N}_\delta(c) = \{c': d(c,c') \le \delta\}$ under metric $d$, define the effective sample size available to estimate $\theta(c)$ by

$$
N_{\text{eff}}(c,\delta)
= \sum_{i=1}^n w_\delta(c_i,c),
\quad
w_\delta(c_i,c) \propto K\!\left(\frac{d(c_i,c)}{\delta}\right),
\quad
\sum_i w_\delta(c_i,c)=1,
$$

where $K$ is a kernel.  
A kernel-regularized estimator with smoothness penalty

$$
\mathcal{R}(\theta) = \int \|\nabla_c \theta(c)\|^2\,\mathrm{d}c
$$

solves

$$
\widehat{\theta}
= \arg\min_{\theta \in \Theta}
\frac{1}{n} \sum_{i=1}^n \ell(\theta; x_i, y_i, c_i)
+ \lambda\, \mathcal{R}(\theta).
$$

Assuming local Lipschitzness in $c$ and $L$-smooth, $\mu$-strongly convex risk in $\theta$, a standard bias–variance decomposition yields for each component $j$

$$
\mathbb{E}\!\left[\|\widehat{\theta}j(c) - \theta_j(c)\|^2\right]
\lesssim
\underbrace{\frac{\sigma^2}{N_{\text{eff}}(c,\delta)}}_{\text{variance}}
+ \underbrace{\delta^{2\alpha}}_{\text{approx. bias}}
+ \underbrace{\lambda^2}_{\text{reg. bias}},
\quad \alpha > 0,
$$

which exhibits the adaptivity–data trade-off: finer locality (small $\delta$) increases resolution but reduces $N_{\text{eff}}$, inflating variance.  
Practical procedures pick $\delta$ and $\lambda$ to balance these terms (e.g., via validation), and amortized approaches replace $\theta(c)$ by $f_\phi(c)$ with shared parameters $\phi$ to increase $N_{\text{eff}}$ through parameter sharing.
For computation, an early-stopped first-order method with step size $\eta$ and $T(c)$ context-dependent iterations satisfies (for smooth, strongly convex risk) the bound

$$
\mathcal{L}\!\big(\theta^{(T(c))}\big) - \mathcal{L}\!\big(\theta^{\star}\big)
\;\le\;
(1-\eta\mu)^{T(c)}\!\left(\mathcal{L}\!\big(\theta^{(0)}\big) - \mathcal{L}\!\big(\theta^{\star}\big)\right)
+ \frac{\eta L \sigma^2}{2\mu\,N_{\text{eff}}(c,\delta)}\,.
$$

linking compute allocation $T(c)$ and data availability $N_{\text{eff}}(c,\delta)$ to the attainable excess risk at context $c$.

### Formal optimization view of context-aware efficiency

Let $f_\phi : \mathcal{X} \times \mathcal{C} \to \mathcal{Y}$ be a context-conditioned predictor with shared parameters $\phi$.  
Given per-context compute budgets $T(c)$ and a global regularizer $\Omega(\phi)$, a resource-aware training objective is

$$
\min_{\phi}\;
\mathbb{E}_{(x,y,c)\sim \mathcal{D}}
\ell\big(f_\phi(x,c),y\big)
+ \lambda\,\Omega(\phi)
\quad
\text{s.t.}\quad
\mathbb{E}_{c}\,\mathcal{C}\big(f_\phi; T(c), c\big) \le B,
$$

where $\mathcal{C}(\cdot)$ models compute or latency.  
The Lagrangian relaxation is

$$
\min_{\phi}\;
\mathbb{E}_{(x,y,c)}
\ell\big(f_\phi(x,c),y\big)
+ \lambda\,\Omega(\phi)
+ \gamma\,\mathbb{E}_{c}\,\mathcal{C}\big(f_\phi; T(c), c\big),
$$

which trades off accuracy and compute via $\gamma$.  
For mixture-of-experts or sparsity-inducing designs, let $\phi = (\phi_1, \ldots, \phi_M)$ and define a gating function $\pi_\phi(m \mid c)$.  
A compute-aware sparsity penalty can be written as

$$
\Omega(\phi)
= \sum_{m=1}^M \alpha_m\,\|\phi_m\|_2^2
+ \tau\, \mathbb{E}_{c} \sum_{m=1}^M \pi_\phi(m\mid c),
$$

encouraging few active modules per context.  
Under smoothness and strong convexity, the optimality conditions yield the KKT stationarity conditions

$$
\nabla_\phi \Big( \mathbb{E}\,\ell
+ \lambda\,\Omega
+ \gamma\,\mathbb{E}_c\,\mathcal{C} \Big)
= 0,
\quad
\gamma\,\big( \mathbb{E}_c\,\mathcal{C} - B \big) = 0,
\quad
\gamma \ge 0.
$$

This perspective clarifies that context-aware efficiency arises from jointly selecting representation sharing, per-context compute allocation $T(c)$, and sparsity in active submodules subject to resource budgets.

Together, these efficiency principles and formal analyses bridge conceptual foundations with implementation. In the next section, we turn to explicit adaptive models that instantiate these ideas through structured parameterization and estimation.


## Explicit Adaptivity: Structured Estimation of $f(c)$

In classical statistical modeling, all observations are typically assumed to share a common set of parameters. However, modern datasets often display significant heterogeneity across individuals, locations, or experimental conditions, making this assumption unrealistic in many real-world applications. To better capture such heterogeneity, recent approaches model parameters as explicit functions of observed context, formalized as $\theta_i = f(c_i)$, where $f$ maps each context to a sample-specific parameter [@doi:10.1111/j.2517-6161.1993.tb01939.x].

A familiar example of explicit adaptivity is multi-task learning, where context is defined by task identity. 
Traditional multi-task learning (left) assigns each task its own head on top of shared representations, 
while context-flagged models (right) pass task identity directly as an input, enabling richer parameter sharing. 
This illustrates how explicit conditioning on context variables can unify tasks within a single model and 
provides an intuitive entry point to more general forms of explicit adaptivity (Figure {@fig:mtl-context}).

![Multi-task learning as explicit adaptivity. In traditional MTL (left), each task has its own head on top of shared layers. In context-flagged models (right), the task identity is provided as an input, enabling a shared model to adapt across tasks.](images/mtl_context.png){#fig:mtl-context width="75%"}


This section systematically reviews explicit adaptivity methods, with a focus on structured estimation of $f(c)$. We begin by revisiting classical varying-coefficient models, which provide a conceptual and methodological foundation for modeling context-dependent effects. We then categorize recent advances in explicit adaptivity according to three principal strategies for estimating $f(c)$: (1) smooth nonparametric models that generalize classical techniques, (2) structurally constrained models that incorporate domain-specific knowledge such as spatial or network structure, and (3) learned function approximators that leverage machine learning methods for high-dimensional or complex contexts. Finally, we summarize key theoretical developments and highlight promising directions for future research in this rapidly evolving field.

### Classical Varying-Coefficient Models: A Foundation

Varying-coefficient models (VCMs) are a foundational tool for modeling heterogeneity, as they allow model parameters to vary smoothly with observed context variables [@doi:10.1111/j.2517-6161.1993.tb01939.x; @doi:10.3390/publications13020019]. In their original formulation, the regression coefficients are treated as nonparametric functions of low-dimensional covariates, such as time or age. The standard VCM takes the form

$$
y_i = \sum_{j=1}^{p} \beta_j(c_i) x_{ij} + \varepsilon_i
$$

where each $\beta_j(c)$ is an unknown smooth function, typically estimated using kernel smoothing, local polynomials, or penalized splines.

This approach provides greater flexibility than fixed-coefficient models and is widely used for longitudinal and functional data analysis. The assumption of smoothness makes estimation and theoretical analysis more tractable, but also imposes limitations. Classical VCMs work best when the context is low-dimensional and continuous. They may struggle with abrupt changes, discontinuities, or high-dimensional and structured covariates. In such cases, interpretability and accuracy can be compromised, motivating the development of a variety of modern extensions, which will be discussed in the following sections.

### Advances in Modeling $f(c)$

Recent years have seen substantial progress in the modeling of $f(c)$, the function mapping context to model parameters. These advances can be grouped into three major strategies: (1) smooth non-parametric models that extend classical flexibility; (2) structurally constrained approaches that encode domain knowledge such as spatial or network topology; and (3) high-capacity machine learning methods for high-dimensional, unstructured contexts. Each strategy addresses specific challenges in modeling heterogeneity, and together they provide a comprehensive toolkit for explicit adaptivity.

#### Smooth Non-parametric Models

This family of models generalizes the classical VCM by expressing $f(c)$ as a flexible, smooth function estimated with basis expansions and regularization. Common approaches include spline-based methods, local polynomial regression, and RKHS-based frameworks. For instance, developed a semi-nonparametric VCM using RKHS techniques for imaging genetics, enabling the model to capture complex nonlinear effects. Such methods are central to generalized additive models, supporting both flexibility and interpretability. Theoretical work has shown that penalized splines and kernel methods offer strong statistical guarantees in moderate dimensions, although computational cost and overfitting can become issues as the dimension of $c$ increases. These estimators occupy the lower-capacity but more interpretable end of the explicit adaptivity spectrum, forming a conceptual baseline for more complex architectures discussed below.

#### Structured Regularization for Graphical and Network Models

The origins of structurally constrained models can be traced to early work on covariance selection. Dempster (1972) demonstrated that zeros in the inverse covariance matrix correspond directly to conditional independencies, introducing the principle that sparsity reflects structure [@doi:10.2307/2528966]. This principle was formalized in Lauritzen’s (1996) influential monograph, which systematized probabilistic graphical models and showed how independence assumptions can be embedded into estimation procedures [@doi:10.1093/oso/9780198522195.001.0001]. Together, these works established the conceptual foundation that explicit structure can guide inference in high-dimensional settings.

As high-dimensional data became common, scalable estimation procedures emerged to make these ideas practical. Meinshausen and Bühlmann (2006) proposed neighborhood selection, recasting graph recovery as a series of sparse regression problems that infer conditional dependencies node by node [@doi:10.1214/009053606000000281]. Shortly thereafter, Friedman, Hastie, and Tibshirani (2008) developed the graphical lasso, a convex penalized likelihood method that directly estimates sparse precision matrices [@doi:10.1093/biostatistics/kxm045]. These contributions showed that sparsity-inducing penalties could recover large network structures reliably, thereby providing concrete tools for estimating $f(c)$ when context corresponds to a structured dependency pattern such as a graph.

Building on these advances, later research recognized that networks themselves may vary across contexts. Guo, Levina, Michailidis, and Zhu (2011) introduced penalties that jointly estimate multiple graphical models, encouraging sparsity within each network while borrowing strength across related groups [@doi:10.1093/biomet/asq060]. Danaher, Wang, and Witten (2014) extended this framework with the Joint Graphical Lasso, which balances shared structure and context-specific edges across multiple populations [@doi:10.1111/rssb.12033]. These developments illustrate how structured regularization transforms explicit adaptivity into a principled strategy: instead of estimating networks independently, one can pool information selectively across contexts (where context $c$ is the group or task identity), making the estimation of the parameter function $f(c)$ both interpretable and statistically efficient.

**Piecewise-Constant and Partition-Based Models.**
Here, model parameters are allowed to remain constant within specific regions or clusters of the context space, rather than vary smoothly. Approaches include classical grouped estimators and modern partition models, which may learn changepoints using regularization tools like total variation penalties or the fused lasso. This framework is particularly effective for data with abrupt transitions or heterogeneous subgroups.

A key design principle is that explicit splits of the context space can emulate distinct tasks, clarifying where parameters should be shared or separated. By introducing hierarchical partitions, we can capture heterogeneity at multiple levels: sample-level variation within each context, and task-level switching across contexts. This perspective connects classical partition-based models with multi-task learning, highlighting how explicit splits of context define where parameters should be shared versus differentiated (Figure {@fig:context-splits}).

![Hierarchical splits of context enable multi-level adaptivity. Explicit adaptivity can partition 
the context space into piecewise models, with parameters indexed both by context $c$ and task 
identity $(i,j)$. Such splits allow sample-level heterogeneity to be captured within contexts, 
while high-level partitions mimic task boundaries and enable task switching.](images/context_splits.png){#fig:context-splits width="75%"}


A subtle but important point is that the boundary between “parametric” and “nonparametric” adaptivity is porous. 
If we fit **simple parametric models within each context** -- for observed contexts $c$ or latent subcontexts $Z$ -- and then **aggregate across contexts**, the resulting conditional

$$
P(Y\mid X,C) \;=\; \int P(Y\mid X,C,Z)\, dP(Z\mid C)
$$

can display rich, multimodal behavior that looks nonparametric. In other words, **global flexibility can emerge from compositional, context-specific parametrics**. 
When component families are identifiable (or suitably regularized) and the context-to-mixture map is constrained (e.g., smoothness/TV/sparsity over $c$), the aggregate model remains estimable and interpretable while avoiding overflexible, ill-posed mixtures.

![Compositional inference: nonparametric flexibility from parametric context-specific models. 
(A) Overall conditional $P(Y \mid X, C)$. 
(B) Context-specific components $P(Y \mid X, C, Z=z_i)$ for latent subgroups $Z$. 
(C) Recombination via marginalization $\int_Z P(Y \mid X, C, Z)$. 
(D) Aggregated distribution showing how structured parametric pieces yield multimodal, nonparametric-like behavior.](images/compositional-inference.png){#fig:compositional-inference width="85%"}

This perspective motivates flexible function approximators: trees and neural networks can be read as learning either the **context-to-mixture weights** or **local parametric maps**, providing similar global flexibility with different inductive biases.

**Structured Regularization for Spatial, Graph, and Network Data.**
When context has known spatial or network structure, regularization terms can promote similarity among neighboring coefficients or nodes. For example, spatially varying-coefficient models have been applied to problems in geographical analysis and econometrics, where local effects are expected to vary across adjacent regions [@doi:10.48550/arXiv.2410.07229; @doi:10.48550/arXiv.2502.14651]. On networked data, the network VCM of [@doi:10.1080/01621459.2025.2470481] generalizes these ideas by learning both the latent positions and the parameter functions on graphs, allowing the model to accommodate complex relational heterogeneity. Such structural constraints allow models to leverage domain knowledge, improving efficiency and interpretability where smooth models may struggle. These regularization principles can also be extended to temporal, hierarchical, or multilevel contexts, where smooth transitions or cross-level coupling may be encoded through Laplacian penalties or nested-group regularizers tailored to the structure of $c$.

Beyond spatial and single-network constraints, Bayesian approaches allow explicit modeling of multiple related graphical models across contexts. Rather than estimating each network independently or pooling across all data, these methods place structured priors that encourage information sharing when appropriate. For example, [@doi:10.1198/jasa.2011.tm10465] introduced Bayesian inference for GGMs with lattice structure, demonstrating how spatial priors can capture context-dependence across neighboring sites. Building on this idea, [@doi:10.1080/01621459.2014.896806]  proposed a Bayesian framework with a Markov random field prior and spike-and-slab formulation to learn when edges should be shared across sample groups, improving estimation and quantifying inter-context similarity. More recently, [@doi:10.1093/biomtc/ujaf053] extended these principles to covariate-dependent graph learning, where network structure varies smoothly with observed covariates. Their dual group spike-and-slab prior enables multi-level selection at node, covariate, and local levels, providing a flexible and interpretable framework for heterogeneous biological networks. Together, these advances illustrate how Bayesian structural priors make adaptivity explicit in graphical models, supporting both efficient estimation and scientific interpretability.

#### Learned Function Approximators

As context dimensionality and data complexity grow, explicit smoothness assumptions become insufficient, motivating high-capacity learners that approximate $f(c)$ directly from data. A third class of methods is rooted in modern machine learning, leveraging high-capacity models to approximate $f(c)$ directly from data. These approaches are especially valuable when the context is high-dimensional or unstructured, where classical assumptions may no longer be sufficient.

**Tree-Based Ensembles.**
Gradient boosting decision trees (GBDTs) and related ensemble methods are well suited to tabular and mixed-type data. A representative example is Tree Boosted Varying-Coefficient Models, introduced by Zhou and Hooker (2019), where GBDTs are applied to estimate context-dependent coefficient functions within a VCM framework [@doi:10.48550/arXiv.1904.01058]. This approach offers a useful balance among flexibility, predictive accuracy, and interpretability, while typically being easier to train and tune than deep neural networks. More recently, Zakrisson and Lindholm (2024) proposed a tree-based varying coefficient model that incorporates cyclic gradient boosting machines (CGBM). Their method enables dimension-wise early stopping and provides feature importance measures, thereby enhancing interpretability and offering additional regularization [@doi:10.48550/arXiv.2401.05982].

Overall, tree-based VCMs achieve strong predictive performance and retain a model structure that lends itself to interpretation, particularly when combined with tools such as SHAP for explaining model outputs.

**Deep Neural Networks.**
For contexts defined by complex, high-dimensional features such as images, text, or sequential data, deep neural networks offer unique advantages for modeling $f(c)$. These architectures can learn adaptive, data-driven representations that capture intricate relationships beyond the scope of classical models. Applications include personalized medicine, natural language processing, and behavioral science, where outcomes may depend on subtle or latent features of the context.

The decision between these machine learning approaches depends on the specific characteristics of the data, the priority placed on interpretability, and computational considerations. Collectively, these advances have significantly broadened the scope of explicit adaptivity, making it feasible to model heterogeneity in ever more complex settings.

### Key Theoretical Advances

The expanding landscape of varying-coefficient models (VCMs) has been supported by substantial theoretical progress, which secures the validity of flexible modeling strategies and guides their practical use. The nature of these theoretical results often reflects the core structural assumptions of each model class.

**Theory for Smooth Non-parametric Models.**
For classical VCMs based on kernel smoothing, local polynomial estimation, or penalized splines, extensive theoretical work has characterized their convergence rates and statistical efficiency. Under standard regularity conditions, these estimators are known to achieve minimax optimality for function estimation in moderate dimensions [@doi:10.1111/j.2517-6161.1993.tb01939.x]. More specifically, Lu, Zhang, and Zhu (2008) established both consistency and asymptotic normality for penalized spline estimators when using a sufficient number of knots and appropriate penalty terms [@doi:10.1080/03610920801931887], enabling valid inference through confidence intervals and hypothesis testing. These results provide a solid theoretical foundation even in relatively complex modeling contexts.

**Theory for Structurally Constrained Models.**
When discrete or network structure is incorporated into VCMs, theoretical analysis focuses on identifiability, regularization properties, and conditions for consistent estimation. For example, [@doi:10.1080/01621459.2025.2470481] provide non-asymptotic error bounds for estimators in network VCMs, demonstrating that consistency can be attained when the underlying graph topology satisfies certain connectivity properties. In piecewise-constant and partition-based models, results from change-point analysis and total variation regularization guarantee that abrupt parameter changes can be recovered accurately under suitable sparsity and signal strength conditions.

**Theory for High-Capacity and Learned Models.**
The incorporation of machine learning models into VCMs introduces new theoretical challenges. For high-dimensional and sparse settings, oracle inequalities and penalized likelihood theory establish conditions for consistent variable selection and accurate estimation, as seen in methods based on boosting and other regularization techniques. In the context of neural network-based VCMs, the theory is still developing, with current research focused on understanding generalization properties and identifiability in non-convex optimization. This remains an active and important frontier for both statistical and machine learning communities.

These theoretical advances provide a rigorous foundation for explicit adaptivity, a wide range of complex and structured modeling scenarios.

### Sparsity and Incomplete Measurements as Context

A central practical challenge in combining real-world datasets is inconsistent measurement: different cohorts or institutions often collect different subsets of features. One dataset may contain detailed laboratory values, another may focus on imaging or physiological measurements, and a third may emphasize clinical outcomes. If such cohorts are naively pooled, the resulting feature matrix is sparse and unbalanced. If incomplete samples are discarded, data efficiency collapses.  

Context-adaptive models provide a natural resolution by treating **measurement sparsity itself as context.** Rather than ignoring missingness, the model learns to adjust its parameterization according to which features are observed. In effect, each measurement policy (labs-only, vitals-only, multimodal) defines a context, and explicit adaptivity allows estimation that respects these differences while still sharing information. This perspective reframes missingness from a nuisance into structured signal: it encodes which sources of evidence are available and how they should be combined. This perspective reframes missingness from a nuisance into structured signal: it encodes which sources of evidence are available and how they should be combined, reflecting ideas explored in recent multimodal learning frameworks that handle missing modalities [@doi:10.48550/arXiv.2409.07825].

![Patterns of missingness as context. Each dataset (e.g., cohort with labs, cohort with vitals, cohort with imaging) provides a different subset of measurements. Context-adaptive models allow integration by conditioning on measurement availability, enabling learning from fewer samples with more heterogeneous features.](images/measurement-sparsity-context.png){#fig:sparsity-context width="70%"}

Figure @fig:sparsity-context illustrates this idea: each cohort contributes a different subset of measurements (lungs, labs, vitals), and explicit adaptivity enables integration across cohorts. By conditioning on measurement availability, we can achieve greater sample efficiency, learning from fewer individuals but with richer heterogeneous features.  

Evaluation of missingness-as-context models should report *mask-stratified metrics*, including worst-group performance, following group-robust evaluation practice [@doi:10.48550/arXiv.1911.08731; @doi:10.48550/arXiv.2012.07421]. Robustness should be probed with *mask-shift stress tests*, training under one measurement policy and testing under another, to quantify degradation and the benefit of contextualization, as formalized in the Domain Adaptation under Missingness Shift (DAMS) setting [@doi:10.48550/arXiv.2211.02093; @doi:10.48550/arXiv.2012.07421]. When imputation is used, authors should assess *imputation realism* by holding out observed entries under realistic mask distributions and reporting MAE/RMSE and calibration for $p(x_{\text{missing}}\mid x_{\text{observed}})$ [@doi:10.48550/arXiv.1806.02382; @doi:10.48550/arXiv.1806.02920]. For causal or estimation applications, conduct *ignorability sensitivity analyses*, contrasting MAR-based results with pattern-mixture or selection-model analyses under plausible MNAR mechanisms [@doi:10.2307/2337120; @doi:10.48550/arXiv.2301.05043]. Finally, include *ablations* that remove mask/indicator inputs—and, for trees, disable default-direction routing—to confirm that gains derive from modeling the mask signal rather than artifacts [@doi:10.48550/arXiv.1603.02754; @doi:10.48550/arXiv.2211.09259]. Practical implementations of these ideas are widely available: **GRU-D** [@doi:10.48550/arXiv.1606.01865] and **BRITS** [@doi:10.48550/arXiv.1805.10572] provide mask- and time-aware sequence models, while **GAIN** [@doi:10.48550/arXiv.1806.02920] and **VAEAC** [@doi:10.48550/arXiv.1806.02382] offer open-source code for imputation under arbitrary masks. For tree ensembles, **XGBoost** supports sparsity-aware default-direction splits, making it straightforward to treat “NA” values as context without preprocessing [@doi:10.1145/2939672.2939785].

### Context-Aware Efficiency Principles and Design

The efficiency of context-adaptive methods hinges on several key design principles that balance computational tractability with statistical accuracy. These principles guide the development of methods that can scale to large datasets while maintaining interpretability and robustness.

One central principle is the use of sparsity assumptions to limit the number of context-dependent parameters. This can be achieved through group sparsity, which encourages entire groups of parameters to be zero simultaneously [@doi:10.1111/j.1467-9868.2005.00532.x], hierarchical regularization that applies different strengths of shrinkage to varying levels of context specificity [@doi:10.1017/CBO9780511790942], and adaptive thresholding that dynamically adjusts sparsity levels in accordance with context complexity.

Efficiency can also be enhanced through computational strategies that allocate resources adaptively. Early stopping terminates optimization for contexts where convergence occurs rapidly [@doi:10.48550/arXiv.1606.04838], while context-dependent sampling employs different sampling schemes across contexts [@doi:10.48550/arXiv.1809.09582]. Caching and warm-starting further accelerate optimization by leveraging solutions from similar contexts, particularly effective when contexts exhibit smooth variation [@doi:10.1561/2200000016].

A further consideration is the balance between efficiency and interpretability. Linear context functions are highly interpretable but may require many parameters, while explicit context encodings improve transparency at the potential cost of higher computational overhead. Local context modeling provides fine-grained interpretability but may be less scalable to large applications. These trade-offs should be evaluated in light of application-specific requirements. For example, advanced adaptive optimizers like Adam can efficiently train complex, nonlinear models, but the resulting systems may be less interpretable than simpler alternatives [@doi:10.48550/arXiv.1412.6980]. In practice, such context-dependent computation appears in adaptive batching, per-context learning rates, and multi-fidelity optimization pipelines that dynamically adjust compute and precision depending on context complexity.

### Synthesis and Future Directions

Selecting an appropriate modeling strategy for $f(c)$ involves weighing flexibility, interpretability, computational cost, and the extent of available domain knowledge. Learned function approximators, such as deep neural networks, offer unmatched capacity for modeling complex, high-dimensional relationships. However, classical smooth models and structurally constrained approaches often provide greater interpretability, transparency, and statistical efficiency. The choice of prior assumptions and the scalability of the estimation procedure are also central considerations in applied contexts.

Looking forward, several trends are shaping the field. One important direction is the integration of varying-coefficient models with foundation models from natural language processing and computer vision. By using pre-trained embeddings as context variables $c_i$, it becomes possible to incorporate large amounts of prior knowledge and extend VCMs to multi-modal and unstructured data sources. Another active area concerns the principled combination of cross-modal contexts, bringing together information from text, images, and structured covariates within a unified VCM framework.

Advances in interpretability and visualization for high-dimensional or black-box coefficient functions are equally important. Developing tools that allow users to understand and trust model outputs is critical for the adoption of VCMs in sensitive areas such as healthcare and policy analysis.

Finally, closing the gap between methodological innovation and practical deployment remains a priority. Although the literature has produced many powerful variants of VCMs, practical adoption is often limited by the availability of software and the clarity of methodological guidance [@doi:10.3390/publications13020019]. Continued investment in user-friendly implementations, open-source libraries, and empirical benchmarks will facilitate broader adoption and greater impact.

In summary, explicit adaptivity through structured estimation of $f(c)$ now forms a core paradigm at the interface of statistical modeling and machine learning. Future progress will focus not only on expanding the expressive power of these models, but also on making them more accessible, interpretable, and practically useful in real-world applications.


## Implicit Adaptivity: Emergent Contextualization in Complex Models

**Introduction: From Explicit to Implicit Adaptivity.**

Traditional models often describe how parameters change by directly specifying a function of context, for example through expressions like $\theta_i = f(c_i)$, where the link between context $c_i$ and parameters $\theta_i$ is fully explicit. In contrast, many modern machine learning systems adapt in fundamentally different ways. Large neural network architectures, particularly foundation models that are now central to state-of-the-art AI research [@doi:10.48550/arXiv.2108.07258], exhibit forms of adaptation that do not arise from any predefined mapping. Instead, their flexibility emerges from the interaction between model structure and the breadth of training data—an effect we refer to as implicit adaptivity. They show a capacity for adaptation that does not arise from any predefined mapping. Instead, their flexibility emerges naturally from the structure of the model and the breadth of the data seen during training. This phenomenon is known as implicit adaptivity. This emergent phenomenon, referred to as implicit adaptivity, highlights how learning and inference can become intertwined within the model itself. Attention 

Unlike explicit approaches, implicit adaptivity does not depend on directly mapping context to model parameters, nor does it always require context to be formally defined. Such models, by training on large and diverse datasets, internalize broad statistical regularities. As a result, they often display context-sensitive behavior at inference time, even when the notion of context is only implicit or distributed across the input. This capacity for emergent adaptation is especially prominent in foundation models, which can generalize to new tasks and domains without parameter updates, relying solely on the information provided within the input or prompt.

In this section, we offer a systematic review of the mechanisms underlying implicit adaptation. We first discuss the core architectural principles that support context-aware computation in neural networks. Next, we examine how meta-learning frameworks deliberately promote adaptation across diverse tasks. Finally, we focus on the advanced phenomenon of in-context learning in foundation models, which highlights the frontiers of implicit adaptivity in modern machine learning. Through this progression, we aim to clarify the foundations and significance of implicit adaptivity for current and future AI systems.

### Foundations of Implicit Adaptation

The capacity for implicit adaptation does not originate from a single mechanism, but reflects a range of capabilities grounded in fundamental principles of neural network design. Unlike approaches that adjust parameters by directly mapping context to coefficients, implicit adaptation emerges from the way information is processed within a model, even when the global parameters remain fixed. To provide a basis for understanding more advanced forms of adaptation, such as in-context learning, this section reviews the architectural components that enable context-aware computation. We begin with simple context-as-input models and then discuss the more dynamic forms of conditioning enabled by attention mechanisms.

#### Architectural Conditioning via Context Inputs

In contrast to explicit parameter mapping, the simplest route to implicit adaptation is to feed context directly as part of the input. The simplest form of implicit adaptation appears in neural network models that directly incorporate context as part of their input. In models written as $y_i = g([x_i, c_i]; \Phi)$, context features $c_i$ are concatenated with the primary features $x_i$, and the mapping $g$ is determined by a single set of fixed global weights $\Phi$. Even though these parameters do not change during inference, the network’s nonlinear structure allows it to capture complex interactions. As a result, the relationship between $x_i$ and $y_i$ can vary depending on the specific value of $c_i$.

This basic yet powerful principle is central to many conditional prediction tasks. For example, personalized recommendation systems often combine a user embedding (as context) with item features to predict ratings. Similarly, in multi-task learning frameworks, shared networks learn representations conditioned on task or environment identifiers, which allows a single model to solve multiple related problems [@doi:10.48550/arXiv.1706.05098].

#### Interaction Effects and Attention Mechanisms

Modern architectures go beyond simple input concatenation by introducing interaction layers that support richer context dependence. These can include feature-wise multiplications, gating modules, or context-dependent normalization. Among these innovations, the attention mechanism stands out as the foundation of the Transformer architecture [@doi:10.48550/arXiv.1706.03762].

Attention allows a model to assign varying degrees of importance to different parts of an input sequence, depending on the overall context. In the self-attention mechanism, each element in a sequence computes a set of query, key, and value vectors. The model then evaluates the relevance of each element to every other element, and these relevance scores determine a weighted sum of the value vectors. This process enables the model to focus on the most relevant contextual information for each step in computation. The ability to adapt processing dynamically in this way is not dictated by explicit parameter functions, but emerges from the network’s internal organization. By enabling dynamic, input-dependent weighting, attention supports context-aware computation without altering global parameters, thereby setting the stage for advanced on-the-fly adaptation such as in-context learning.

### Amortized Inference and Meta-Learning

Moving beyond fixed architectures that implicitly adapt, another family of methods deliberately trains models to become efficient learners. These approaches, broadly termed meta-learning or "learning to learn," distribute the cost of adaptation across a diverse training phase. As a result, models can make rapid, task-specific adjustments during inference. Rather than focusing on solving a single problem, these methods train models to learn the process of problem-solving itself. This perspective provides an important conceptual foundation for understanding the in-context learning capabilities of foundation models.

#### Amortized Inference

Amortized inference represents a more systematic form of implicit adaptation. In this setting, a model learns a reusable function that enables rapid inference for new data points, effectively distributing the computational cost over the training phase. In traditional Bayesian inference, calculating the posterior distribution for each new data point is computationally demanding. Amortized inference addresses this challenge by training an "inference network" to approximate these calculations. A classic example is the encoder in a Variational Autoencoder (VAE), which is optimized to map high-dimensional observations directly to the parameters, such as mean and variance, of an approximate posterior distribution over a latent space [@doi:10.48550/arXiv.1312.6114]. The inference network thus learns a complex, black-box mapping from the data context to distributional parameters. Once learned, this mapping can be efficiently applied to any new input at test time, providing a fast feed-forward approximation to a traditionally costly inference process.

#### Meta-Learning: Learning to Learn

Meta-learning builds upon these ideas by training models on a broad distribution of related tasks. The explicit goal is to enable efficient adaptation to new tasks. Instead of optimizing performance for any single task, meta-learning focuses on developing a transferable adaptation strategy or a parameter initialization that supports rapid learning in novel settings [@doi:10.48550/arXiv.2004.05439].

Gradient-based meta-learning frameworks such as Model-Agnostic Meta-Learning (MAML) illustrate this principle. In these frameworks, the model discovers a set of initial parameters that can be quickly adapted to a new task with only a small number of gradient updates [@doi:10.48550/arXiv.1703.03400]. Training proceeds in a nested loop: the inner loop simulates adaptation to individual tasks, while the outer loop updates the initial parameters to improve adaptability across tasks. As a result, the capacity for adaptation becomes encoded in the meta-learned parameters themselves. When confronted with a new task at inference, the model can rapidly achieve strong performance using just a few examples, without the need for a hand-crafted mapping from context to parameters. In this view, the capacity to adapt becomes encoded in the meta-learned parameters themselves, enabling rapid generalization from few examples without a hand-crafted map from context to coefficients and standing in clear contrast to explicit approaches.

### In-Context Learning in Foundation Models

The most powerful and, arguably, most enigmatic form of implicit adaptivity is in-context learning (ICL), an emergent capability of large-scale foundation models. This phenomenon has become a central focus of modern AI research, as it represents a significant shift in how models learn and adapt to new tasks. This section provides an expanded review of ICL, beginning with a description of the core phenomenon, then deconstructing the key factors that influence its performance, reviewing the leading hypotheses for its underlying mechanisms, and concluding with its current limitations and open questions.

#### The Phenomenon of Few-Shot In-Context Learning

First systematically demonstrated in large language models such as GPT-3 [@doi:10.48550/arXiv.2005.14165], ICL is the ability of a model to perform a new task after being conditioned on just a few examples provided in its input prompt. Critically, this adaptation occurs entirely within a single forward pass, without any updates to the model's weights. For instance, a model can be prompted with a few English-to-French translation pairs and then successfully translate a new word, effectively learning the task on the fly. This capability supports a broad range of applications, including few-shot classification, following complex instructions, and even inducing and applying simple algorithms from examples. Subsequent work has shown that the ability to generalize from few in-context examples can itself be enhanced through meta-training. MetaICL explicitly trains models across diverse meta-tasks, teaching them to infer and adapt within context at test time without gradient updates, thereby strengthening the implicit adaptability of large language models [@doi:10.48550/arXiv.2110.15943].

#### Deconstructing ICL: Key Influencing Factors

The effectiveness of ICL is not guaranteed and depends heavily on several interacting factors, which have been the subject of extensive empirical investigation.

**The Role of Scale.**
A critical finding is that ICL is an emergent ability that appears only after a model surpasses a certain threshold in scale (in terms of parameters, data, and computation). Recent work has shown that larger models do not just improve quantitatively at ICL; they may also learn in qualitatively different ways, suggesting that scale enables a fundamental shift in capability rather than a simple performance boost [@doi:10.48550/arXiv.2206.07682].

**Prompt Engineering and Example Selection.**
The performance of ICL is highly sensitive to the composition of the prompt. The format, order, and selection of the in-context examples can dramatically affect the model's output. Counterintuitively, research has shown that the distribution of the input examples, rather than the correctness of their labels, often matters more for effective ICL. This suggests that the model is primarily learning a task format or an input-output mapping from the provided examples, rather than learning the underlying concepts from the labels themselves [@doi:10.48550/arXiv.2202.12837].

#### Hypothesized Mechanisms: How Does ICL Work?

The underlying mechanisms that enable ICL are not fully understood and remain an active area of research. Several leading hypotheses have emerged, viewing ICL through the lenses of meta-learning, Bayesian inference, and specific architectural components.

**ICL as Implicit Meta-Learning.**
The most prominent theory posits that transformers learn to implement general-purpose learning algorithms within their forward pass. During pre-training on vast and diverse datasets, the model is exposed to a multitude of tasks and patterns. This process is thought to implicitly train the model as a meta-learner, allowing it to recognize abstract task structures within a prompt and then execute a learned optimization process on the provided examples to solve the task for a new query [@doi:10.48550/arXiv.2212.10559; @doi:10.48550/arXiv.2308.16898].

**ICL as Implicit Bayesian Inference.**
A complementary and powerful perspective understands ICL as a form of implicit Bayesian inference. In this view, the model learns a broad prior over a large class of functions during its pre-training phase. The in-context examples provided in the prompt act as evidence, which the model uses to perform a Bayesian update, resulting in a posterior predictive distribution for the final query. This framework provides a compelling explanation for how models can generalize from very few examples [@doi:10.48550/arXiv.2111.02080]. A complementary theoretical development interprets in-context learning as a rational adaptation process. From a Bayesian decision-theoretic standpoint, transformers can be viewed as implicitly balancing expected loss with strategy complexity, thereby achieving near-optimal adaptation under computational constraints [@doi:10.48550/arXiv.2506.17859]. This rational framing connects implicit adaptivity with classical principles of statistical inference.

**The Role of Induction Heads.**
From a more mechanistic, architectural perspective, researchers have identified specific attention head patterns, dubbed "induction heads," that appear to be crucial for ICL. These specialized heads are hypothesized to form circuits that can scan the context for repeated patterns and then copy or complete them, providing a basic mechanism for pattern completion and generalization from in-context examples [@doi:10.48550/arXiv.2209.11895]. Extending this mechanistic line, Dherin et al. (2025) demonstrate that stacking self-attention and MLP layers allows transformers to implicitly update internal representations during a single forward pass, effectively realizing dynamic context-specific weight adjustments without explicit training [@doi:10.48550/arXiv.2507.16003]. Such implicit internal updates offer a concrete mechanistic account of how context-dependent behavior arises.

#### Limitations and Open Questions

Despite its remarkable capabilities, ICL faces significant limitations with respect to transparency, explicit control, and robustness. The adaptation process is opaque, making it difficult to debug or predict failure modes. Furthermore, performance can be brittle and highly sensitive to small changes in the prompt. As summarized in recent surveys, key open questions include developing a more complete theoretical understanding of ICL, improving its reliability, and establishing methods for controlling its behavior in high-stakes applications [@doi:10.48550/arXiv.2301.00234].

### Theoretical Bridges Between Varying-Coefficient Models and In-Context Learning

Recent theoretical work has uncovered deep connections between classical varying-coefficient models and the mechanisms underlying in-context learning in transformers. 
Although these approaches arise from different traditions — one grounded in semi-parametric statistics, the other in large-scale deep learning — they can implement strikingly similar estimators. 
This section formalizes these parallels and reviews key theoretical results establishing these bridges.

#### Varying-Coefficient Models as Kernel Regression

Consider a semi-parametric varying-coefficient model in which each observation is governed by a parameter vector $\theta_i$ that depends smoothly on context $c_i$. 
For a new query context $c^\ast$, the parameter estimate is obtained by solving a locally weighted likelihood problem:

$$
\widehat{\theta}(c^\ast) 
= \arg\max_{\theta} \sum_{i=1}^n K_\lambda(c_i, c^\ast)\,\ell(x_i; \theta),
$$

where $K_\lambda$ is a kernel function that measures similarity between contexts and $\ell$ is the log-likelihood.

For regression with squared loss, this reduces to kernel ridge regression in the context space. 
Let $y = (y_1,\dots,y_n)^\top$ and $K \in \mathbb{R}^{n \times n}$ be the Gram matrix with $K_{ij} = k(c_i, c_j)$. 
The prediction at $c^\ast$ is

$$
\widehat{y}(c^\ast) = k(c^\ast)^\top (K + \lambda I)^{-1} y,
$$

where $k(c^\ast) = (k(c^\ast, c_1), \ldots, k(c^\ast, c_n))^\top$. 
This expression highlights that varying-coefficient models perform kernel smoothing in the context space: nearby observations in context have greater influence on the parameter estimates at $c^\ast$.

Equivalently, the fitted model can be written as

$$
\widehat{f}(x^\ast, c^\ast) = \sum_{i=1}^n \alpha_i(c^\ast)\, y_i,
$$

where $\alpha_i(c^\ast)$ are normalized kernel weights determined entirely by the context similarities and the regularization parameter $\lambda$.

#### Transformers as Ridge and Kernel Regressors In-Context

A parallel line of research has demonstrated that transformers trained on simple regression tasks can learn to perform ridge or kernel regression entirely within their forward pass, without any explicit supervision to do so.

Akyürek et al. (2022) show that for linear regression tasks, transformers can learn to implement the ridge regression estimator

$$
\widehat{w} = (X^\top X + \lambda I)^{-1} X^\top y
$$

directly from a sequence of in-context examples. Each example $(x_i, y_i)$ is represented as a token, and the query token attends to the support tokens to compute the prediction for $x^\ast$; the attention mechanism learns to encode the solution to the regression problem [@doi:10.48550/arXiv.2211.15661].

Building on this finding, von Oswald et al. (2023) show that gradient-based training of transformers over distributions of regression tasks leads them to perform in-context gradient descent, effectively realizing kernel regression with the learned attention kernel serving as $k(c_i, c_j)$ [@doi:10.48550/arXiv.2212.07677]. Garg et al. (2023) further analyze which function classes can be learned in-context, demonstrating that transformers can approximate a wide family of kernel smoothers when trained on synthetic regression tasks [@doi:10.48550/arXiv.2208.01066].

Dai et al. (2023) provide a complementary theoretical view, arguing that transformers can implicitly implement compositional function families through their attention layers, and that in-context learning arises naturally from this functional representation [@doi:10.48550/arXiv.2212.10559].

Finally, Reuter et al. (2025) propose a compelling Bayesian interpretation: transformers trained under in-context learning can perform full Bayesian inference for common statistical models such as generalized linear models and latent factor models. Concretely, they train transformers to infer complex posterior distributions in context, showing that the in-context forward pass can approximate posterior sampling comparable to MCMC or variational inference methods [@doi:10.48550/arXiv.2501.16825].

In all these cases, the support set within the prompt plays an analogous role to the neighborhood in context space in varying-coefficient models. The query token’s prediction is formed by aggregating information from the support tokens via learned similarity weights, realized by the attention mechanism rather than an explicitly defined kernel function.

#### Synthesis: Two Paths to the Same Estimators

Taken together, these results reveal a common form:

$$
\widehat{f}(x^\ast, c^\ast) = \sum_{i=1}^n \alpha_i(c^\ast)\, y_i,
$$

where the weights $\alpha_i(c^\ast)$ depend on the relationship between the query context $c^\ast$ and support contexts $\{c_i\}$.

- In varying-coefficient models, $\alpha_i(c^\ast)$ are determined explicitly by a user-chosen kernel $K_\lambda$.
- In transformers, $\alpha_i(c^\ast)$ emerge implicitly from the learned attention patterns and internal computations after pretraining.

Both perspectives yield estimators of the same functional form, with explicit kernel weighting in VCMs and learned attention weighting in transformers. This correspondence motivates a unified view of context-adaptive inference, combining the interpretability of explicit modeling with the flexibility and scale of implicit computation. This bridge motivates a unified framework for studying context-adaptive inference: explicit methods provide interpretability and structure, while implicit methods provide flexibility and scalability. Understanding how these two meet offers a promising path toward adaptive, interpretable models at scale. This unified perspective is also extending to structured and tabular domains. TabICL introduces a foundation model architecture for large-scale tabular data, showing that in-context learning can efficiently scale to structured datasets via column-row attention mechanisms [@doi:10.48550/arXiv.2502.05564]. These results suggest that implicit adaptivity generalizes beyond text or vision into the broader landscape of structured scientific data.


### Comparative Synthesis: Implicit versus Explicit Adaptivity

Implicit and explicit strategies reflect two complementary philosophies for modeling heterogeneity, each with distinct strengths and trade-offs. The optimal choice between these approaches depends on the goals of analysis, the structure and scale of available data, and the need for interpretability or regulatory compliance in the application domain.

**Implicit Adaptivity.**
The principal advantage of implicit methods lies in their remarkable flexibility and scalability. Leveraging large-scale pre-training on diverse datasets, these models can effectively adapt to high-dimensional and unstructured contexts, such as raw text, images, or other complex sensory data, where explicitly specifying a context function $f(c)$ is infeasible. Because adaptation is performed internally during the model’s forward pass, inference is both rapid and adaptable. However, the mechanisms underlying this adaptability are typically opaque, making it challenging to interpret or control the model’s decision process. In applications like healthcare or autonomous systems, this lack of transparency can hinder trust, validation, and responsible deployment.

**Explicit Adaptivity.**
In contrast, explicit models provide direct, interpretable mappings from context to parameters through functions such as $f(c)$. This structure supports clear visualization, statistical analysis, and the formulation of scientific hypotheses. It also enables more direct scrutiny and control of the model’s reasoning. Nevertheless, explicit methods rely heavily on domain expertise to specify an appropriate functional form, and may struggle to accommodate unstructured or highly complex context spaces. If the assumed structure is misspecified, the model’s performance and generalizability can be severely limited.

In summary, these two paradigms illustrate a fundamental trade-off between expressive capacity and transparent reasoning. Practitioners should carefully weigh these considerations, often choosing or blending approaches based on the unique demands of the task. For clarity, a comparative table or figure can further highlight the strengths and limitations of each strategy across various real-world applications.

### Open Challenges and the Motivation for Interpretability

The rise of powerful implicit adaptation methods, particularly in-context learning, raises critical open research questions regarding their diagnosis, control, and reliability. As these models are deployed in increasingly high-stakes applications, understanding their failure modes is not just an academic exercise but a practical necessity [@doi:10.48550/arXiv.2108.07258]. It is important to develop systematic methods for assessing when and why in-context learning is likely to fail, and to create techniques for interpreting and, where possible, steering the adaptation process. Prompting strategies such as chain-of-thought demonstrate that structured context can sometimes steer internal computation, providing limited but useful handles on model behavior [@doi:10.48550/arXiv.2201.11903]. A thorough understanding of the theoretical limits and practical capabilities of implicit adaptivity remains a central topic for ongoing research.

These considerations motivate a growing search for techniques that can make the adaptation process more transparent by "making the implicit explicit." Such methods aim to bridge the gap between the powerful but opaque capabilities of implicit models and the need for trustworthy, reliable AI. This research can be broadly categorized into several areas, including post-hoc interpretability approaches that seek to explain individual predictions [@doi:10.3390/e23010018], surrogate modeling where a simpler, interpretable model is trained to mimic the complex model's behavior, and strategies for extracting modular structure from trained models. A prime example of the latter is the line of work probing language models to determine if they have learned factual knowledge in a structured, accessible way [@doi:10.48550/arXiv.1909.01066]. By surfacing the latent structure inside these systems, researchers can enhance trust, promote modularity, and improve the readiness of adaptive models for deployment in real-world settings. This line of work provides a conceptual transition to subsequent sections, which explore the integration of interpretability with adaptive modeling.


## Toward Explicit Modeling of Implicit Adaptivity: Local Models, Surrogates and Post Hoc Approximations

### Motivation
Building on the prior discussion of implicit adaptivity, this section examines methods that expose, approximate, or control those adaptive mechanisms.  
Implicit adaptivity allows powerful models, including foundation models, to adjust behavior without explicitly representing a mapping from context to parameters [@doi:10.48550/arXiv.2108.07258]. This flexibility obscures the underlying mechanisms of adaptation, hindering modular reuse and systematic auditing. Making adaptivity explicit improves alignment with downstream goals, enables modular composition, and supports debugging and error attribution. It also fits the call for a more rigorous science of interpretability with defined objectives and evaluation criteria [@doi:10.48550/arXiv.1702.08608; @doi:10.48550/arXiv.2402.02870].  
This chapter reviews practical approaches for surfacing structure, the assumptions they rely on, and how to evaluate their faithfulness and utility.


**From Implicit to Explicit Adaptivity**  
Implicit adaptivity is hidden, flexible, and hard to audit, while explicit adaptivity surfaces modular structure that is structured, auditable, and controllable. The transition highlights three key trade-offs developed in this section: **Fidelity vs. Interpretability**, **Local vs. Global Scope**, and **Approximation vs. Control**.  

![From Implicit to Explicit Adaptivity. A black-box model (left) represents implicit adaptation, which is hidden and opaque. Making adaptivity explicit (right) exposes structured components that can be inspected and controlled. The axes below highlight the trade-offs between fidelity and interpretability, local and global scope, and approximation and control.](images/explicit_from_implicit.png){#fig:implicit-to-explicit width="80%"}  


### Approaches
Efforts to make implicit adaptation explicit span complementary strategies that differ in assumptions, granularity, and computational cost. We group them into six families:

1. surrogate modeling for local approximation,  
2. prototype- and neighbor-based reasoning,  
3. diagnostics for amortized inference,  
4. disentanglement and bottleneck methods,  
5. parameter extraction and probing, and  
6. emerging approaches that leverage large language models as post-hoc explainers.

#### Surrogate Modeling
This line of work approximates a black-box $h(x,c)$ with an interpretable model in a small neighborhood, so that local behavior and a local view of $f(c)$ can be inspected. A formal template is

$$
\hat{g}_{x_0,c_0} = \arg\min_{g \in \mathcal{G}} \, \mathbb{E}_{(x,c) \sim \mathcal{N}_{x_0,c_0}} \left[ \ell\big(h(x,c), g(x,c)\big) \right] + \Omega(g),
$$

where $\mathcal N_{x_0,c_0}$ defines a locality (e.g., kernel weights), $\ell$ measures fidelity, and $\Omega$ controls complexity. Where $\mathcal{G}$ denotes a restricted hypothesis class, often composed of linear or other low-complexity functions chosen to enhance interpretability. A convenient local goodness-of-fit is

$$
R^2_{\text{local}}
= 1 - \frac{\sum_i w_i\,\big(h_i - g_i\big)^2}{\sum_i w_i\,\big(h_i - \bar h\big)^2},
\qquad
w_i \propto \kappa\!\big((x_i,c_i),(x_0,c_0)\big).
$$

LIME perturbs inputs and fits a locality-weighted linear surrogate [@doi:10.48550/arXiv.1602.04938]; SHAP / DeepSHAP provide additive attributions based on Shapley values [@doi:10.48550/arXiv.1705.07874]. Integrated Gradients and DeepLIFT link attribution to path-integrated sensitivity or reference-based contributions [@doi:10.48550/arXiv.1703.01365; @doi:10.48550/arXiv.1704.02685]. These methods are most reliable when the model is near-linear in the chosen neighborhood and perturbations remain near the data manifold; consequently, a rigorous analysis involves stating the neighborhood definition, reporting the surrogate’s goodness-of-fit, and assessing stability across seeds and baselines.

#### Prototype and Nearest-Neighbor Methods
Here, a decision is grounded by reference to similar cases in representation space, which supports case-based explanations and modular updates. ProtoPNet learns a library of visual prototypes to implement “this looks like that” reasoning [@doi:10.48550/arXiv.1806.10574]. Deep $k$-nearest neighbors audits predictions by querying neighbors in activation space and can flag distribution shift [@doi:10.48550/arXiv.1803.04765]. Influence functions link a prediction to influential training points for data-centric debugging [@doi:10.48550/arXiv.1703.04730]. This line of work connects naturally to exemplar models and contextual bandits, where decisions are justified via comparisons to context-matched exemplars. Reports include prototype coverage and diversity, neighbor quality checks, and the effect of editing prototypes or influential examples. These prototype-based approaches make local adaptation explicit by grounding predictions in reference cases, bridging the gap between black-box models and case-based reasoning frameworks.

#### Amortization Diagnostics
For amortized inference systems (e.g., VAEs), the encoder $q_{\phi}(\theta\mid x)$ can be treated as an implicit $f(c)$. Diagnostics measure amortization gaps and identify suboptimal inference or collapse [@doi:10.48550/arXiv.1801.03558]. Useful outputs include calibration under shift and posterior predictive checks, together with ablations that vary encoder capacity or add limited iterative refinement. This clarifies when the learned mapping is faithful versus when it underfits the target posterior. Such diagnostics mirror classical checks for approximate Bayesian inference, where amortization gaps quantify the discrepancy between learned and exact posteriors.

#### Disentangled and Bottlenecked Representations
While amortization diagnostics target model faithfulness, disentanglement aims to expose interpretable subspaces aligned with distinct contextual factors. The aim is to expose factors that align with distinct contextual causes, making changes traceable and controllable. $\beta$-VAE encourages more factorized latents [@higgins2017betavae], while the Deep Variational Information Bottleneck promotes predictive minimality that can suppress spurious context [@doi:10.48550/arXiv.1612.00410]. Concept-based methods such as TCAV and ACE map latent directions to human concepts and test sensitivity at the concept level [@doi:10.48550/arXiv.1711.11279; @doi:10.48550/arXiv.1902.03129]. Fully unsupervised disentanglement is often ill-posed without inductive bias or weak supervision [@doi:10.48550/arXiv.1811.12359]. Quantitative evaluation of disentanglement can follow established metrics that assess factor independence, completeness, and informativeness [@eastwood2018a]. Reports should include concept validity tests, factor stability across runs, and simple interventions that demonstrate controllability.

#### Parameter Extraction and Probing
This family locates where adaptation is encoded and exposes handles for inspection or edits. Linear probes test what is linearly decodable from intermediate layers [@doi:10.48550/arXiv.1610.01644]; edge probing examines specific linguistic structure in contextualized representations [@doi:10.48550/arXiv.1905.06316]. Model editing methods such as ROME can modify stored factual associations directly in weights [@doi:10.48550/arXiv.2202.05262], while “knowledge neurons” seek units linked to particular facts [@doi:10.48550/arXiv.2104.08696]. Evaluation involves quantifying pre- and post-edit behavior, assessing locality and persistence, and documenting side effects on unrelated capabilities. Collectively, these methods transform hidden internal adaptations into analyzable modular components.

#### LLMs as Post-hoc Explainers
Recent work uses in-context prompting to elicit rationales, counterfactuals, or error hypotheses from large language models for a target system [@doi:10.48550/arXiv.2310.05797]. These explanations can be useful but must be validated for faithfulness, for example by checking agreement with surrogate attributions, reproducing input–output behavior, and testing stability to prompt variations. Explanations should be treated as statistical estimators with stated objectives and evaluation criteria [@doi:10.48550/arXiv.2402.02870].

These methodological families differ in their assumptions and computational granularity, yet they all aim to render adaptation transparent and controllable. The following sections summarize their key trade-offs and conceptual challenges.

### Trade-offs

#### Fidelity vs. Interpretability
High-fidelity surrogates capture the target model’s behavior more accurately, yet they often grow in complexity and lose readability. A crisp statement of the design goal is

$$
\min_{g\in\mathcal G}\ \underbrace{\phi_{\text{fid}}(g;U)}_{\text{faithfulness on use set }U}
+ \lambda\\underbrace{\psi_{\text{simplicity}}(g)}_{\text{sparsity / size / semantic load}},
$$

where $\phi_{\text{fid}}$ can be local $R^2$, AUC, or rank correlation with $h$, and $\psi_{\text{simplicity}}$ can be sparsity, tree depth, rule count, or active concept count. If a simple surrogate underfits, consider structured regularization (e.g., monotonic constraints, grouped sparsity, concept bottlenecks). If a complex surrogate is needed, accompany it with readable summaries (partial dependence snapshots, distilled rule sets, compact concept reports).

#### Local vs. Global Scope
Local surrogates aim for $g_{x_0,c_0}\approx h$ only on $\mathcal N_{x_0,c_0}$, whereas a global surrogate seeks $g_{\text{global}}\approx h$ across the domain, potentially smoothing away distinct regimes. Hybrid schemes combine both:

$$
g(x,c)=\sum_{k=1}^{K} w_k(x,c)\, g_k(x,c),
\qquad \sum_k w_k(x,c)=1,\quad w_k\ge 0,
$$

with local experts $g_k$ and soft assignment $w_k$. Report the neighborhood definition, coverage (fraction of test cases with acceptable local fit), and disagreements between local and global views; flag regions where the global surrogate is unreliable.

#### Approximation vs. Control
Coarse modularization makes control and auditing simpler because edits act on a small number of levers, yet residual error can be large. Fine-grained extraction, such as neuron- or weight-level edits, can achieve precise behavioral changes but may introduce unintended side effects. Define the intended edit surface in advance (concepts, features, prototypes, submodules, parameters). For coarse modules, measure the residual gap to the base model and verify that edits improve target behavior without harming unaffected cases. For fine-grained edits, quantify locality and collateral effects using a held-out audit suite with counterfactuals, canary tasks, and out-of-distribution probes. Maintain versioned edits, enable rollback, and document the scope of validity.

These trade-offs are not merely design choices but determine the operational boundaries within which explicit representations can remain faithful to the original adaptive system.

### Open Research Directions

#### Reusable Modules
The challenge of isolating reusable routines parallels the quest for parameter-efficient fine-tuning in large models, where adaptation must remain modular yet composable. A central question is whether we can isolate portable skills or routines from large models and reuse them across tasks without degrading overall capability [@doi:10.48550/arXiv.2108.07258]. Concretely, a reusable module should satisfy portability, isolation, composability, and stability. Promising directions include concept bottlenecks that expose human-aligned interfaces, prototype libraries as swappable reference sets, sparse adapters that confine changes to limited parameter subsets, and routing mechanisms that select modules based on context. Evaluation should track transfer performance, sample efficiency, interference on held-out capabilities, and robustness under domain shift.

#### Performance Gains
When does making structure explicit improve robustness or efficiency compared to purely implicit adaptation? Benefits are most likely when domain priors are reliable, data are scarce, or safety constraints limit free-form behavior. Explicit structure is promising when context topology is known (spatial or graph), when spurious correlations should be suppressed, and when explanations must be auditable. To assess this, fix capacity and training budget and vary only the explicit structure (prototypes, disentanglement, bottlenecks). Stress tests should cover diverse distributional challenges, including covariate shift, concept shift, long-tail classes, and adversarially correlated features. Account for costs such as concept annotation, extra hyperparameters, and potential in-domain accuracy loss.

#### Abstraction Level
Another open issue is the appropriate level at which to represent structure: parameters (weights, neurons), functions (local surrogates, concept scorers, routing policies), or latent causes (disentangled or causal factors). Benchmarking under fixed capacity and identical data regimes is essential to isolate the contribution of explicit structure from mere model scaling effects. Choose based on the use case. For safety patches, lower-level handles allow precise edits but require guardrails and monitoring. For scientific or policy communication, function- or concept-level interfaces are often more stable and auditable. Optimize three objectives in tension: faithfulness to the underlying model, usability for the target audience, and stability under shift. Tooling should support movement between levels (e.g., distilling weight-level edits into concept summaries or lifting local surrogates into compact global reports). Selecting the proper level of abstraction thus defines not only interpretability but also the feasible scope of control.

### Evaluation and Reporting Standards for Classical Post-hoc Methods
LIME, SHAP, and gradient-based methods such as Integrated Gradients and DeepLIFT remain common tools for context-adaptive interpretation. Their usefulness depends on careful design and transparent reporting. Explanations should be treated as statistical estimators with stated objectives and evaluation criteria [@doi:10.48550/arXiv.1702.08608; @doi:10.48550/arXiv.2402.02870]. Carmichael & Scheirer (2021) further propose a principled evaluation framework for feature-additive explainers, enabling measurement of misattribution even under known ground-truth additive models [@doi:10.48550/arXiv.2106.08376].

#### Scope and locality
Local surrogate methods require a clear definition of the neighborhood in which the explanation is valid. The sampling scheme, kernel width, and surrogate capacity determine which aspects of the black box can be recovered. When context variables are present, the explanation should be conditioned on the relevant context and the valid region should be described.

#### Attribution methods in practice
Attribution based on gradients is sensitive to baseline selection, path construction, input scaling, and preprocessing. Baselines should have clear domain meaning, and sensitivity analyses should show how conclusions change under alternative baselines. For perturbation-based surrogates, report the perturbation distribution and any constraints that keep samples on the data manifold.

#### Faithfulness and robustness
Faithfulness and robustness should be checked rather than assumed. Useful checks include deletion and insertion curves, counterfactual tests, randomization tests, stability under small input and seed perturbations, and for local surrogates a local goodness-of-fit such as a neighborhood $R^2$. The evaluation metric should match the stated objective of the explanation [@doi:10.48550/arXiv.1702.08608; @doi:10.48550/arXiv.2402.02870]. Turbé et al. (2023) demonstrate evaluation of interpretability methods on time-series models using metrics such as $\widetilde{\mathrm{AUC}}_{S}$ and $\widetilde{F_{1,S}}$ to compare alignment with model internals [@doi:10.1038/s42256-023-00620-w].  

#### Minimal reporting checklist
| Item | Description |
|------|--------------|
| **Data slice and context definition** | Specify the subset of data and contextual variables used for generating explanations, and describe the locality or neighborhood definition. |
| **Surrogate specification and regularization details** | Report the family of surrogate models, chosen regularization strategy, and kernel or sampling parameters. |
| **Faithfulness and robustness metrics** | Include local $R^2$, deletion/insertion area, counterfactual validity, and robustness under perturbations. |
| **Sensitivity and uncertainty analysis** | Assess variation across baselines, random seeds, and small input perturbations, providing uncertainty estimates. |
| **Computational constraints** | Document runtime, hardware limitations, and approximation budgets that affect explanation quality. |
| **Observed limitations and failure modes** | Summarize known weaknesses, unstable regions, or interpretability failures identified during validation. |

**Table 2. Minimal Reporting Checklist for Post-hoc Explanations**

#### From post hoc analysis to design
Insights from post-hoc analysis can inform proactive model design for control, auditing, and policy communication. In such cases, interpretability methods should not remain external diagnostics but serve as guides for architectures with built-in transparency. For example, Concept Bottleneck Models integrate interpretable concepts into the forward pass [@doi:10.48550/arXiv.2007.04612]. Similarly, Poursabzi-Sangdeh et al. (2021) conduct empirical user studies to highlight how interpretability design choices affect human use and model trust [@doi:10.48550/arXiv.1802.07810]. These contributions extend the vision of Doshi-Velez & Kim (2017) toward a unified science of interpretable modeling, where explanation and model training are co-designed [@doi:10.48550/arXiv.1702.08608]. Taken together, these lines of work bridge black-box adaptation and structured inference and set the stage for designs where context-to-parameter mappings are specified, trained, and evaluated end to end.

#### Implications for classical models
These tools can also clarify how traditional models, for example, logistic regression with interaction terms or generalized additive models to admit a local adaptation view: a simple global form paired with context-sensitive weights or features. Reading such models through the lens of local surrogates and concept interfaces helps align classical estimation with modern, context-adaptive practice. Reinterpreting these classical estimators through the lens of explicit adaptivity situates them as early instances of structured context modeling, underscoring continuity between statistical modeling and modern machine learning. 

Taken together, these strategies illustrate a gradual unification of interpretability, modularization, and adaptive modeling, paving the way toward a principled science of explicit context-aware inference.



## Context-Invariant Training: A View from the Converse

While the preceding sections emphasize the importance of modeling context to tailor predictions, an equally fundamental question concerns robustness: Can we learn representations such that a single predictor performs reliably across sites, cohorts, and time, despite environmental shifts and nuisance variation? Context-invariant training aims at out-of-distribution (OOD) generalization by emphasizing features whose associations with the target remain stable across environments, while suppressing spurious correlations that vary with nuisance contexts. Standard Empirical Risk Minimization (ERM) [@vapnik1991principles] often latches onto spurious, environment-specific correlations. In practice, this means using multiple environments during training and favoring representations that make a single readout perform well everywhere. 

The seminal framework connecting modern deep learning to invariant prediction is Invariant Risk Minimization (IRM) [@arXiv:1907.02893], which formulates robustness as learning causally stable predictors across multiple environments. IRM seeks a representation $\Phi$ such that a shared predictor $w$ minimizes the risk $R^e(\cdot)$ for every environment $e$. The original formulation is a bi-level optimization problem that is computationally intractable. To make it solvable, Arjovsky et al. propose a surrogate version, IRMv1, which introduces a penalty ensuring that the per-environment risk gradient vanishes for a shared dummy classifier $w=1$, thereby enforcing stationarity across environments. This construction connects invariance to out-of-distribution (OOD) generalization by encouraging predictors aligned with causal mechanisms that persist across environments.

However, subsequent analyses revealed important limitations. In linear settings, IRM often fails to recover the true invariant predictor, and in nonlinear regimes, performance can deteriorate sharply when test distributions deviate from the training domains [@arXiv:2010.05761]. This undermines IRM’s objective of handling distribution shift, where $P(X)$ changes while $P(Y|X)$ remains fixed. Thus, IRM offers no mechanism to reduce sensitivity when those shifts are amplified at test time. To mitigate these issues, Risk Extrapolation (REx) [@arXiv:2003.00688] extends the principle of invariance by optimizing directly over per-environment risk vectors. Two practical variants have been proposed: MM-REx and V-REx, which performs robust optimization over affine combinations of the environment risks (weights sum to 1, possibly negative), and V-REx, which minimizes the mean risk augmented by the variance of risks across environments.

Unlike IRM, which requires explicit environment labels, Beery et al. (2018) [@arXiv:1710.11469] propose CoRe, a method that assumes some samples share a common identifier. Features are decomposed into core components (whose class-conditional distribution is stable across domains) and style components (e.g., brightness, pose) that vary with domains. The CoRe estimator enforces robustness by penalizing the conditional variance of the loss within groups sharing the same label–identifier pair $(Y, ID)$.


### Adversarial Robustness as Context-Invariant Training
Whereas IRM seeks robustness across discrete environments, adversarial robustness can be regarded as its infinitesimal counterpart—focusing on perturbations within a local neighborhood of each input rather than across distinct domains. Those different environments can be interpreted as fine-grained, synthetic perturbations around each data point rather than distinct real-world domains.
Invariant learning generally seeks predictors whose performance remains stable when the data-generating context changes — for example, across hospitals, time periods, or demographic groups [@arXiv:1501.01332]. Adversarial robustness follows the same principle of invariance, but at a much finer scale: instead of using naturally occurring environments, it constructs synthetic “environments” through small, deliberate perturbations of the input data. These perturbations simulate local environmental shifts around each sample and expose the model to worst-case contexts.
From this perspective, adversarial robustness is essentially context-invariant learning under infinitesimal, adversarially generated environments.
Each adversarial example $x'=x+\delta$ (where $\|\delta\|_p \le \varepsilon$) can be interpreted as belonging to a neighboring environment of the original input x. Training the model to perform consistently under such local shifts enforces a form of fine-grained invariance that complements the coarse-grained invariance targeted by IRM.
The paper [@arXiv:1706.06083] addresses the vulnerability of deep learning models to adversarial attacks from the optimization view. Specifically, the authors interpret adversarial robustness as a min-max optimization problem, where the goal is to minimize the worst-case loss incurred by adversarial examples. Madry et al. (2018) introduce Projected Gradient Descent (PGD) as a universal first-order adversary. The generated perturbations are incorporated into the training process to improve robustness under local contextual shifts. In this view, the environments in IRM correspond to multiple data domains, while those in adversarial training correspond to local neighborhoods of each sample—both formulations share the same objective of minimizing performance variation across shifts in context. Formally, both IRM and adversarial training minimize performance variance across contexts—IRM across discrete environments, and adversarial training across continuous perturbation neighborhoods.

[@arXiv:1805.12152] provably demonstrates the trade-off between robustness and accuracy in machine learning models. The authors argue that adversarial training, while improving robustness to adversarial perturbations, can decrease the model's accuracy on clean data. This occurs because adversarial training forces the model to adjust its decision boundaries, which may lead to a loss in standard performance. The paper also shows that the representations learned by robust models align better with salient data characteristics and human perception, which suggests that robust models focus more on features that are meaningful and interpretable. At the same time, robust models tend to learn representations that align better with salient data characteristics and human perception, suggesting that robustness promotes the extraction of stable, semantically meaningful features, mirroring the goal of context-invariant learning at a smaller, instance-specific scale [@arXiv:1905.02175].

This perspective is directly applicable to the challenges faced by LLM-based Agents as surveyed in [@arXiv:2309.07864]. An autonomous agent does not operate in a sterile, curated dataset; it operates in the wild. These fine-grained, synthetic perturbations provide a useful abstraction for understanding the robustness challenges faced by LLM-based agents:

**Perception Robustness**: A small, imperceptible change to an image or a document an agent is analyzing (an adversarial perturbation) could cause it to completely misinterpret its environment and take a disastrous action.

**Tool-Use Robustness**: A slight rephrasing of a user's command could trick a non-robust agent into generating incorrect or malicious code for a tool to execute.
<!-- 
Related references:

- Towards Deep Learning Models Resistant to Adversarial Attacks [@arXiv:1706.06083]
- Robustness May Be at Odds with Accuracy [@arXiv:1805.12152]

- The Rise and Potential of Large Language Model Based Agents: A Survey [@arXiv:2309.07864] -->

Hence, advances in adversarial robustness directly inform the design of safer, more context-stable autonomous agents.

### Training methods for Context-Invariant Models
While the principle of context-invariance is a powerful theoretical goal, several practical training methodologies have been developed to approximate it, primarily by enhancing robustness against group shifts. These methods vary in their assumptions, particularly regarding the availability of explicit group or environment labels for the training data.

A foundational approach, applicable when group labels are available, is Group Distributionally Robust Optimization (Group DRO). Unlike standard Empirical Risk Minimization (ERM) which minimizes the average loss over the entire dataset, formulated as:
$$
\min_{f} \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i)
$$
Group DRO's objective is to minimize the loss on the worst-performing data group. This is formally expressed as a min-max problem:
$$
\min_{f} \max_{g \in \mathcal{G}} \mathbb{E}_{(x,y) \sim P_g} [L(f(x), y)]
$$
where $\mathcal{G}$ represents the set of all predefined groups and $P_g$ is the data distribution for a specific group $g$ [@arXiv:1911.08731]. However, the authors identify a critical pitfall: in modern, overparameterized neural networks, this method can fail. Such models can easily memorize the entire training set, reducing the worst-case training loss to zero without actually learning a generalizable solution. The key insight from this work is that **strong regularization** (such as a high L2 penalty or aggressive early stopping) is essential. Regularization prevents the model from perfectly fitting the training data, forcing it to learn simpler, more robust features that generalize better to the worst-case groups on unseen data.
The primary limitation of Group DRO is its reliance on fully annotated training data, a luxury seldom available in real-world scenarios. This challenge has spurred the development of methods that operate without explicit group information. These approaches cleverly leverage the inherent biases of standard models as a source of information.
A simple and highly effective heuristic is Just Train Twice (JTT) [@arXiv:2107.09044]. This method operates in two stages: first, a standard ERM model is trained for several epochs. Second, the training examples that this initial model misclassified are identified and upweighted. A new model is then trained from scratch on this reweighted dataset. The underlying assumption is that a standard model's errors serve as an effective proxy for identifying examples from minority or difficult groups. By focusing the second stage of training on these hard examples, JTT improves worst-group performance without ever needing to know the group labels.
Providing a more formalized framework, Environment Inference for Invariant Learning (EIIL) aims to bridge the gap between unlabeled data and invariant learning algorithms like IRM [@arXiv:2010.07249]. Similar to JTT, EIIL begins by training a standard ERM model. It then uses the biases of this reference model to automatically partition the dataset into several inferred "environments." For instance, examples the model confidently gets right might form one environment, while those it gets wrong form another. These algorithmically generated environment labels can then be fed into any off-the-shelf invariant learning method to train a final, robust model. EIIL essentially transforms the problem from one requiring manual labels to one where environments can be discovered directly from the data itself.
Collectively, these approaches demonstrate a continuum from fully supervised environment-aware optimization to self-supervised environment discovery, unified under the goal of achieving context-invariant generalization. Together, these methods illustrate a clear progression from fully-supervised techniques to more practical approaches that cleverly infer hidden data structure, all aiming to build models that are more robust and invariant to challenging shifts in context.



<!-- - Just Train Twice: Improving Group Robustness without Training Group Information [@arXiv:2002.10384]
- Environment Inference for Invariant Learning [@arXiv:2110.14048]
- Distributionally Robust Neural Networks for Group Shifts [@arXiv:1911.08731] -->



## Applications, Case Studies, Evaluation Metrics, and Tools

This section surveys how context-adaptive methods manifest across domains, how their performance is assessed, and what tools enable them in practice.

### Implementation Across Sectors

Many real-world environments are dynamic and unpredictable, meaning that models built on static assumptions often fail when conditions shift. To remain reliable, models must be able to adapt to changing inputs, contexts, and behaviors. This adaptability is especially important in high-stakes domains where decisions directly affect human well-being or carry significant financial consequences. Two prominent examples are healthcare and finance. In healthcare, context-adaptive models enable more personalized treatment decisions and support early intervention by capturing the evolving state of patients and diseases. In finance, these models capture rapidly changing market conditions, allowing forecasts and risk assessments to remain accurate in volatile times.

Healthcare is one of the domains that benefits greatly from context-aware models because clinical and biomedical data are often hierarchical, exhibiting nested structures and evolving over time. For example, patients may have repeated measurements (e.g., vitals, labs) nested within visits, and these visits are themselves nested within broader care episodes. At the same time, disease trajectories and treatment responses are highly dynamic, requiring models that can adapt to changing contexts rather than assuming static relationships. Several reviews highlight the importance of methods that explicitly account for such complexity in longitudinal and multilevel health data [@doi:10.1002/9780470973394; @doi:10.1177/0962280217706728]. One concrete example is a Bayesian multilevel time-varying joint model that captures complex structures while estimating diverse time-varying relationships, including both response–predictor and response–response dependencies [@doi:10.1002/sim.9582]. Such models often employ hierarchical priors to borrow strength across patients while maintaining individualized inference. In this framework, time-varying coefficients are flexibly estimated using Bayesian P-splines, and inference is performed through Markov Chain Monte Carlo (MCMC). The result is a computationally efficient algorithm that provides interpretable modeling of patient outcomes as they evolve over time.

In finance, context-aware models are particularly valuable for capturing the complex dynamics that unfold both over time and across countries, sectors, and assets, which together drive macroeconomic and market behavior. For instance, cross-sectional dependencies, which capture interconnectedness at the same point in time, emerge when shocks propagate differently across regions or industries, while temporal dependencies, which capture persistence across time, arise from persistent volatility clustering and regime changes. Several reviews and comparative studies emphasize the need for methods that can adapt to such heterogeneity in modern financial data [@doi:10.1016/j.chaos.2008.07.022; @doi:10.48550/arXiv.2212.03471]. A prominent line of work develops Bayesian matrix dynamic factor models (MDFMs), which provide a powerful framework for analyzing matrix-valued time series increasingly common in macro-finance applications [@doi:10.48550/arXiv.2409.08354]. These models incorporate multiple context-adaptive features. On the temporal side, an autoregressive factor process captures persistent comovement and improves recursive forecasting, while stochastic volatility, fat-tailed error distributions, and explicit COVID-19 outlier adjustments allow the model to remain robust under real-world market shocks. The approximate factorization reduces complexity from cubic to linear in the number of assets, making large-scale forecasting feasible.

### Context-Aware Efficiency in Practice

The principles of context-aware efficiency find practical applications across diverse domains, demonstrating how computational and statistical efficiency can be achieved through intelligent context utilization.

In healthcare applications, context-aware efficiency enables adaptive imaging protocols that adjust scan parameters based on patient context such as age, symptoms, and medical history, reducing unnecessary radiation exposure. Personalized screening schedules optimize screening frequency based on individual risk factors and previous results, while resource allocation systems efficiently distribute limited healthcare resources based on patient acuity and context.

Financial services leverage context-aware efficiency principles in risk assessment by adapting risk models based on market conditions, economic indicators, and individual borrower characteristics. Fraud detection systems use context-dependent thresholds and sampling strategies to balance detection accuracy with computational cost, while portfolio optimization dynamically adjusts rebalancing based on volatility regimes and transaction costs, as studied in regime-switching portfolio models [@doi:10.1093/rfs/15.4.1137].

Industrial applications benefit from context-aware efficiency through predictive maintenance systems that adapt maintenance schedules based on equipment context including age, usage patterns, and environmental conditions [@doi:10.1109/TR.2016.2570568]. Quality control implements context-dependent sampling strategies that focus computational resources on high-risk production batches, and inventory management uses context-aware forecasting to optimize stock levels across different product categories and market conditions.

A notable example of context-aware efficiency is adaptive clinical trial design, where trial parameters are dynamically adjusted based on accumulating evidence while maintaining statistical validity. Population enrichment refines patient selection criteria based on early trial results, and dose finding optimizes treatment dosages based on individual patient responses and safety profiles. These applications demonstrate how context-aware efficiency principles can lead to substantial improvements in both computational performance and real-world outcomes.

### Formal Metrics for Evaluating Context-Aware Performance

Building on the theoretical framework introduced in earlier sections, we now formalize the evaluation criteria used to quantify context-adaptive behavior.  
These metrics capture predictive accuracy, adaptation efficiency, transferability, and robustness under contextual variation.

Let $\mathcal{C}$ denote the context space and $\mathcal{D}_{\mathrm{test}}$ a test distribution over $(x, y, c)$.  
For a predictor $\hat{f}$, define the context-conditional risk as

$$
\mathcal{R}(\hat{f}\mid c)
= \mathbb{E}\!\left[\, \ell\big(\hat{f}(x,c), y\big) \,\middle|\, c \,\right],
\qquad
\mathcal{R}(\hat{f})
= \mathbb{E}_{c\sim \mathcal{D}_{\mathrm{test}}}\!\left[\, \mathcal{R}(\hat{f}\mid c) \,\right].
$$

A context-stratified evaluation reports $\mathcal{R}(\hat{f}\mid c)$ across predefined bins or via a smoothed estimate  
$\int \mathcal{R}(\hat{f}\mid c)\,\mathrm{d}\Pi(c)$ for a reference measure $\Pi$ that weights regions of the context space.

#### Adaptation Efficiency

To evaluate how rapidly a model benefits from in-context examples,  
let $S_k(c)=\{(x_j, y_j, c)\}_{j=1}^k$ denote $k$ examples available within context $c$.  
Define the adaptation efficiency as

$$
\mathrm{AE}_k(c)
= \mathcal{R}(\hat{f}_0 \mid c)
- \mathcal{R}(\hat{f}_{S_k} \mid c),
\qquad
\mathrm{AE}_k
= \mathbb{E}_{c}\!\left[\, \mathrm{AE}_k(c) \,\right],
$$

where $\hat{f}_0$ is the non-adapted baseline and $\hat{f}_{S_k}$ the adapted predictor.  
The function $k \mapsto \mathrm{AE}_k$ summarizes few-shot adaptation gains across different context sizes.

#### Transfer Performance

Transfer across source and target contexts, $\mathcal{C}_{\mathrm{src}} \to \mathcal{C}_{\mathrm{tgt}}$,  
with shared representation $\phi$, can be measured by

$$
\mathrm{TP}(\phi)
= \mathcal{R}_{\mathcal{C}_{\mathrm{tgt}}}\!\big(\hat{f}_{\phi}\big)
- \mathcal{R}_{\mathcal{C}_{\mathrm{tgt}}}\!\big(\hat{f}_{\mathrm{scratch}}\big),
$$

quantifying performance retained when transferring $\phi$ from source to target contexts compared with training from scratch.

#### Robustness to Context Shift

To assess stability under distributional perturbations,  
let $Q$ denote a family of admissible context shifts (e.g., $f$-divergence or Wasserstein balls over context marginals).  
Then the robustness score is defined as

$$
\mathrm{RS}(\hat{f}; Q)
= \sup_{\widetilde{\mathcal{D}}\in Q}
\left[
\mathcal{R}_{\widetilde{\mathcal{D}}}(\hat{f})
- \mathcal{R}_{\mathcal{D}_{\mathrm{test}}}(\hat{f})
\right],
$$

where higher values indicate greater sensitivity to contextual changes.

These metrics provide a unified quantitative view of context-aware performance.  
They complement the theoretical efficiency results developed in Section 4  
and serve as practical diagnostics for evaluating real-world adaptivity across diverse applications.

### Context-Aware Efficiency in Practice

The principles of context-aware efficiency find practical applications across diverse domains, demonstrating how computational and statistical efficiency can be achieved through intelligent context utilization.

In healthcare applications, context-aware efficiency enables adaptive imaging protocols that adjust scan parameters based on patient context such as age, symptoms, and medical history, reducing unnecessary radiation exposure. Personalized screening schedules optimize screening frequency based on individual risk factors and previous results, while resource allocation systems efficiently distribute limited healthcare resources based on patient acuity and context.

Financial services leverage context-aware efficiency principles in risk assessment by adapting risk models based on market conditions, economic indicators, and individual borrower characteristics. Fraud detection systems use context-dependent thresholds and sampling strategies to balance detection accuracy with computational cost, while portfolio optimization dynamically adjusts rebalancing frequency based on market volatility and transaction costs [@doi:10.1109/TR.2016.2570568].

Industrial applications derive clear benefits from context-aware efficiency. In predictive maintenance, systems adapt maintenance schedules using equipment context such as age, usage history, and environmental conditions. For example, recent surveys of predictive maintenance in Industry 4.0 identify architectures that integrate sensor data, remaining-useful-life models, and context-aware scheduling policies [@doi:10.1016/j.cie.2020.106889; @doi:10.1109/WETICE57085.2023.10477842]. In quality control, context-dependent sampling directs inspection efforts to high-risk units, reducing waste and computational cost. Inventory management likewise benefits from context-aware forecasting models that incorporate demand volatility, seasonality, and external signals; recent work shows that such approaches outperform traditional forecasts in retail settings [@doi:10.1080/13675567.2025.2566806].

A notable example of context-aware efficiency is adaptive clinical trial design, where trial parameters are dynamically adjusted based on accumulating evidence while maintaining statistical validity. Population enrichment refines patient selection criteria based on early trial results, and dose finding optimizes treatment dosages based on individual patient responses and safety profiles. These applications demonstrate how context-aware efficiency principles can lead to substantial improvements in both computational performance and real-world outcomes.


### Contextualized Network Inference

One domain where context-adaptive models have shown particular promise is in network inference for genomics. Traditional approaches assume that all samples can be pooled into a single network, or that cohorts can be partitioned into homogeneous groups. These assumptions are often unrealistic: cancer, for example, exhibits both cross-patient heterogeneity and within-patient shifts in gene regulation. 

Contextualized network models address this challenge by learning archetypal networks and then representing each sample as a mixture of these archetypes, weighted by its observed context. This formulation allows researchers to move beyond average-case networks and uncover mechanisms of disease, heterogeneity across patients, driver mutations, and structural hazards.

Such contextualized networks have been applied in TCGA cancer genomics to identify patient-specific driver modules.

![Contextualized networks enable inference of archetypal and sample-specific mixtures, unlocking new biological insights such as mechanisms of disease, disease heterogeneity, structural hazards, and driver mutations.](images/contextualized_networks.png){#fig:contextualized-networks width="90%"}


### Performance Evaluation

Evaluating context-adaptive models requires careful consideration of predictive accuracy, robustness to variability, and scalability, with the emphasis varying by domain. Key aspects of performance evaluation include the choice of metrics, the handling of uncertainty, and assessment under stress or rare-event conditions.

In healthcare, evaluation prioritizes patient-specific predictive accuracy and calibrated uncertainty. Common metrics include mean squared error (MSE), concordance indices (C-index), and calibration curves, which measure how well models capture longitudinal patient trajectories and provide reliable uncertainty estimates. Multi-target Bayesian approaches and survival models demonstrate the importance of capturing correlations across outcomes and assessing credible interval coverage to quantify predictive confidence [@arXiv:2509.08183; @arXiv:2509.01794]. Evaluations in this domain also highlight trade-offs between model complexity, interpretability, and computational feasibility, since high-fidelity patient-level predictions can be costly to compute.

In finance and macro forecasting, performance evaluation emphasizes predictive accuracy under volatile conditions and resilience to structural breaks. Metrics such as root mean squared forecast error (RMSFE), log-likelihood, and stress-test performance are commonly used to assess how well models handle crises or abrupt shifts in data [@arXiv:2409.08354; @arXiv:2508.10055]. Probabilistic metrics, including posterior predictive checks and uncertainty bounds, provide additional insight into the reliability of forecasts, while chaos-informed diagnostics can highlight vulnerabilities to extreme events [@arXiv:2406.12274].

Across domains, consistent patterns emerge. Context-adaptive models outperform static baselines when variability is structured and partially predictable, but performance can degrade in data-sparse regimes or under unmodeled abrupt changes [@arXiv:2303.02781v1]. Evaluations therefore combine error-based measures, probabilistic calibration, and robustness tests to give a holistic view of model performance. The focus should be on these evaluation criteria, rather than the models themselves, to understand where and why context-adaptive approaches provide real advantages. Hence, evaluation protocols must jointly assess accuracy, calibration, and transferability under context perturbations.

### Survey of Tools

There are many technological supports that have emerged to support context-adaptive modeling. These tools provide the infrastructure, memory, and efficiency mechanisms that allow models to operate effectively in dynamic environments.

Retrieval-augmented generation (RAG) has become a core support for adaptivity, enabling models to incorporate new knowledge at inference time instead of relying only on static parameters. Recent surveys outline how RAG architectures combine dense retrievers, re-rankers, and generators into pipelines that continuously update with external information. This allows models to remain aligned with changing knowledge bases [@arXiv:2410.12837]. Beyond improving factuality, RAG also underpins adaptive behavior in AI-generated content, where external retrieval reduces hallucination and provides domain-specific grounding [@arXiv:2402.19473]. These systems depend on efficient vector search. Tools such as FAISS use approximate nearest neighbor algorithms to index billions of embeddings with low latency, while Milvus integrates distributed storage to scale such systems across production environments [@arXiv:1702.08734]. Together, retrieval pipelines and vector databases constitute the infrastructure through which context-adaptive models dynamically expand their accessible knowledge.

While retrieval addresses external knowledge, memory systems support continuity within ongoing interactions. Research on AI memory frameworks emphasizes how models require mechanisms to persist relevant context, get rid of redundancy, and resurface information at appropriate times [@arXiv:2504.15965]. Recent implementations such as MemoryOS illustrate how adaptive memory systems can summarize past conversations, cluster related items, and strategically reinsert them into prompts, producing long-term coherence that can’t be achieved with static context windows alone [@arXiv:2506.06326]. These memory architectures extend adaptivity from the level of just accessing facts to maintaining evolving histories, allowing models to not just adjust to new data, but also to be more consistent and contextually aware of their interactions.

Another critical support lies in scaling sequence length. Standard transformers suffer quadratic complexity and degraded performance as contexts grow, making it difficult to adapt to long or streaming data. New serving infrastructures such as StreamingLLM introduce rolling caches that let models handle long inputs without full recomputation, while frameworks like vLLM use paged attention to manage GPU memory efficiently during extended inference [@arXiv:2309.17453; @arXiv:2309.06180]. This long-context support shifts adaptability from handling snapshots of information to maintaining awareness across evolving information streams.

### Selection and Usage Guidance

Deploying context-adaptive models effectively requires careful alignment between model capabilities, domain needs, and practical constraints.

In healthcare, where data is often hierarchical and time-varying, Bayesian multilevel models and generalized varying-coefficient frameworks are well suited because they can flexibly capture nonlinear interactions and evolving patient trajectories. In finance, high-dimensional time series demand scalability, making matrix dynamic factor models more appropriate than fully specified multivariate systems.

Domain priorities should drive tool choice. Clinical applications often require interpretable models that clinicians can trust, favoring spline-based or single-index approaches even if they sacrifice some predictive accuracy. In contrast, finance applications typically prioritize forecasting performance under volatility, where more complex factor models can offer a competitive edge despite reduced transparency.

Many context-adaptive models rely on resource-intensive inference methods such as MCMC, which may limit scalability. Approximate inference techniques like variational Bayes or stochastic optimization can mitigate this burden for large datasets. In real-time decision settings, long-context processing methods such as StreamingLLM or KV-cache compression provide efficiency gains but require specialized engineering and hardware support.

Finally, tool selection should reflect whether the primary objective is scientific insight or operational decision-making. Biomedical research benefits most from flexible, interpretable models that generate new hypotheses, whereas domains like trading demand models capable of rapid adaptation, scalable inference, and strong predictive accuracy under uncertainty.

There is no one-size-fits-all context-adaptive model. Successful deployment depends not only on technical choices but also on aligning model adaptivity with domain-specific interpretability and governance requirements.


## Future Trends and Opportunities with Foundation Models

### A New Paradigm for Context-Adaptive Inference
Recent advances in large-scale foundation models have fundamentally reshaped the landscape of context-adaptive inference. Trained on vast and diverse datasets with self-supervised objectives, these models internalize broad statistical regularities across language, vision, and multimodal data [@doi:10.48550/arXiv.2108.07258]. Unlike earlier approaches that relied on hand-crafted features or narrowly scoped models, foundation models can process and structure complex, high-dimensional contexts that were previously intractable.

Their impact is clear in natural language processing, where large language models achieve strong zero-shot and few-shot generalization, and in computer vision, where multimodal encoders such as CLIP align images and text into a shared representation space [@doi:10.48550/arXiv.2103.00020]. These advances mark a shift from treating feature extraction and inference as separate stages toward unified systems that function simultaneously as representation learners and adaptive engines. At the same time, challenges remain, including high computational demands, the risk of amplifying societal biases, and the difficulty of interpreting learned representations [@doi:10.1145/3442188.3445922].

To understand their contribution to context-adaptive inference, we consider three dimensions: their role as universal context encoders, the mechanisms enabling dynamic adaptation, and their integration with formal statistical and causal reasoning.

#### Universal Context Encoders
Foundation models act as general-purpose context encoders, transforming raw, unstructured data into meaningful representations without manual feature engineering. For textual data, models such as BERT learn embeddings that capture semantic and syntactic nuances, supporting tasks from classification to retrieval [@doi:10.48550/arXiv.1810.04805]. For visual and multimodal inputs, CLIP aligns images and text into a shared embedding space, enabling zero-shot classification and cross-modal retrieval [@doi:10.48550/arXiv.2103.00020].

These representations effectively serve as context variables—latent, structured features that can feed directly into statistical models. Classical approaches such as regression or causal inference can thus operate on data that would otherwise remain unstructured. This capacity forms the basis for integrating representation learning with formal frameworks of context-adaptive inference.

#### Dynamic Adaptation Mechanisms
Foundation models enable dynamic adaptation primarily at inference time, allowing models to respond to new tasks without retraining. The most prominent mechanism is in-context learning (ICL), where models adapt behavior by conditioning on examples in a prompt, enabling rapid few-shot or zero-shot generalization [@doi:10.48550/arXiv.2208.01066].

Scaling is supported by modular architectures such as Mixture-of-Experts (MoE), which route inputs to specialized sub-networks for sparse activation, increasing capacity without proportional compute [@doi:10.48550/arXiv.1701.06538]. Parameter-efficient fine-tuning (PEFT) methods such as LoRA show that models can be adapted by updating less than one percent of weights, achieving near full fine-tuning performance [@doi:10.48550/arXiv.2106.09685].

Together, these approaches illustrate how adaptation can be achieved both flexibly and efficiently, balancing generalization and computational constraints.

#### Bridging with Statistical and Causal Reasoning
An emerging research direction integrates the representational capacity of foundation models with the rigor of statistical and causal inference. Language models can already extract relational patterns from text to propose or critique causal graphs [@doi:10.48550/arXiv.2305.07171]. Methods such as LMPriors treat foundation models as task-specific priors, improving sample efficiency in estimation and decision making [@doi:10.48550/arXiv.2210.12530]. Models can also generate natural-language rationales that clarify predictions and summarize statistical findings, enhancing interpretability and transparency [@doi:10.48550/arXiv.2310.05797].

Consequently, foundation models serve as bridges between flexible representation learning and principled inference, offering a path toward adaptive systems that are both data-efficient and theoretically grounded.

### Next-Generation Methods for Contextualized Adaptive Inference
While current foundation models already enable impressive forms of adaptivity, the next phase of research looks toward methods that will shape the future of contextualized adaptive inference. These directions point ahead, emphasizing how models may be adapted, combined, and evaluated. The aim is not only greater power, but also more transparency and reliability in high-stakes settings. We highlight three forward-looking methodological trends: modular fine tuning and compositional adaptation, mechanistic insights into in-context learning, and new frameworks for reliability and calibration.

#### Modular Fine-Tuning and Compositional Adaptation
Parameter-efficient fine-tuning approaches such as adapters and LoRA show that large models can be customized by updating only a small subset of parameters while preserving pretrained knowledge [@doi:10.48550/arXiv.2106.09685]. Future systems are expected to expand these ideas into compositional strategies, dynamically combining specialized modules optimized for different domains or contexts [@doi:10.48550/arXiv.2005.00247].

Recent findings suggest that merging multiple LoRA modules can even outperform full fine-tuning, signaling a paradigm where adaptation arises from modular reuse rather than retraining [@doi:10.48550/arXiv.2402.15414]. Compositional adaptation thus points toward building libraries of reusable context-specific skills that can be flexibly assembled for new tasks.

#### In-Context Learning and Mechanistic Insights
Although in-context learning has revolutionized generalization, its internal mechanisms remain partly opaque. Evidence suggests that transformers may implement optimization-like updates during forward passes, effectively performing implicit gradient descent when processing examples [@doi:10.48550/arXiv.2212.07677]. Other analyses interpret ICL as implicit Bayesian inference, where the prompt provides evidence that reshapes the predictive distribution [@doi:10.48550/arXiv.2306.04891].

Mechanistic studies further identify induction heads within transformer attention circuits as critical components for pattern induction and few-shot generalization [@doi:10.48550/arXiv.2209.11895]. Such insights are expected to inspire architectures that enhance both transparency and stability in adaptive learning.

#### Reliability, Calibration, and Context-Sensitive Evaluation
As adaptive models become more flexible, ensuring calibration and reliability across shifting contexts becomes crucial. Deep neural networks, including LLMs, are often miscalibrated, producing overconfident probabilities misaligned with true accuracy [@doi:10.48550/arXiv.1706.04599].

Future research will increasingly embed uncertainty quantification into adaptive pipelines through deep ensembles, Bayesian ensembling, or conformal prediction to produce valid confidence intervals [@doi:10.48550/arXiv.2012.07421]. Evaluation protocols must also stress robustness under distributional shifts, testing whether models can sustain performance and express uncertainty under novel or adversarial conditions [@doi:10.48550/arXiv.2211.09110].

By embedding calibration and robustness within design, adaptive inference can evolve toward a more trustworthy, auditable, and context-aware standard.

### Expanding Frameworks with Foundation Models

Foundation models refer to large-scale, general-purpose neural networks, predominantly transformer-based architectures, trained on vast datasets using self-supervised learning [@doi:10.48550/arXiv.2108.07258]. Their flexibility, scalability, and cross-domain generalization have transformed statistical modeling and data analysis.

LLMs such as GPT-4 [@doi:10.48550/arXiv.2303.08774] and LLaMA-3.1 [@doi:10.48550/arXiv.2407.21783] exemplify this progress, achieving state-of-the-art results in language understanding, summarization, and reasoning. Beyond NLP, foundation models extend to multimodal tasks [@doi:10.48550/arXiv.2103.00020], text embeddings [@doi:10.48550/arXiv.1810.04805], and even tabular and structured data [@doi:10.48550/arXiv.2207.01848].

Adaptivity in these systems is largely realized through prompting, which conditions responses on user-provided context without additional fine-tuning [@doi:10.1145/3560815]. Meanwhile, Mixture-of-Experts (MoE) architectures enhance scalability by routing computation to relevant submodels for efficiency [@doi:10.48550/arXiv.1701.06538].

#### Foundation Models as Context

Foundation models offer significant opportunities by supplying context-aware information that enhances various stages of statistical modeling and inference:

**Feature Extraction and Interpretation:** Foundation models transform raw, unstructured data into structured and interpretable representations. For example, targeted prompts enable LLMs to extract insightful features from text, providing meaningful insights and facilitating interpretability [@doi:10.48550/arXiv.2302.12343, @doi:10.48550/arXiv.2305.12696, @doi:10.18653/v1/2023.emnlp-main.384]. This allows statistical models to operate directly on semantically meaningful features rather than on raw, less interpretable data.

**Contextualized Representations for Downstream Modeling:** Foundation models produce adaptable embeddings and intermediate representations useful as inputs for downstream models, such as decision trees or linear models [@doi:10.48550/arXiv.2208.01066]. These embeddings significantly enhance the training of both complex, black-box models [@doi:10.48550/arXiv.2212.09741] and simpler statistical methods like n-gram-based analyses [@doi:10.1038/s41467-023-43713-1], thereby broadening the application scope and effectiveness of statistical approaches.

**Post-hoc Interpretability:** Foundation models support interpretability by generating natural-language explanations for decisions made by complex models. This capability enhances transparency and trust in statistical inference, providing clear insights into how and why certain predictions or decisions are made [@doi:10.48550/arXiv.2409.08466].

#### Recent Innovations and Outlook
Several new architectures exemplify how foundation models advance context-sensitive inference through modularity and interpretability:

**FLAN-MoE** (Fine-tuned Language Model with Mixture of Experts) [@doi:10.48550/arXiv.2305.14705] combines instruction tuning with expert selection, dynamically activating relevant sub-models based on the context. This method significantly improves performance across diverse NLP tasks, offering superior few-shot and zero-shot capabilities. It also facilitates interpretability through explicit expert activations. Future directions may explore advanced expert-selection techniques and multilingual capabilities.

**LMPriors** (Pre-Trained Language Models as Task-Specific Priors) [@doi:10.48550/arXiv.2210.12530] leverages semantic insights from pre-trained models like GPT-3 to guide tasks such as causal inference, feature selection, and reinforcement learning. This method markedly enhances decision accuracy and efficiency without requiring extensive supervised datasets. However, it necessitates careful prompt engineering to mitigate biases and ethical concerns.

**Mixture of In-Context Experts** (MoICE) [@doi:10.48550/arXiv.2210.12530] introduces a dynamic routing mechanism within attention heads, utilizing multiple Rotary Position Embeddings (RoPE) angles to effectively capture token positions in sequences. MoICE significantly enhances performance on long-context sequences and retrieval-augmented generation tasks by ensuring complete contextual coverage. Efficiency is achieved through selective router training, and interpretability is improved by explicitly visualizing attention distributions, providing detailed insights into the model's reasoning process.

Collectively, these directions suggest a future in which foundation models evolve from passive representation learners into active, context-sensitive inference engines that unify adaptivity, efficiency, and interpretability within a principled framework.


## Open Problems

Rapid advances in context-adaptive modeling have created unprecedented opportunities while revealing fundamental challenges. This chapter identifies the central methodological questions and the broader ethical and societal challenges that will shape the future trajectory of context-adaptive inference. We begin by examining five interrelated technical questions—on modularity, the benefits of explicit structure, the level of abstraction, theoretical and practical barriers, and interpretability trade-offs—that together define the frontier of adaptive modeling research. We then turn to the broader outlook, focusing on the ethical and societal implications of deploying these powerful adaptive systems.

### Open Research Questions

Recent advances have broadened the scope of adaptive inference, but many questions remain unresolved. These open problems span five domains: (i) modularity and reusability of adaptive components, (ii) the conditions under which explicit structure improves robustness and generalization, (iii) the appropriate level of abstraction for intervention, (iv) theoretical, computational, and data-related barriers to adoption, and (v) the tension between interpretable-by-design and post-hoc interpretability. Together, these questions delineate a research agenda that bridges theoretical statistics, machine learning, and applied modeling, combining methodological depth with practical impact.

First, researchers need to examine whether skills and routines can be modularized in a way that allows portability across tasks without interference. Second, the field must clarify under what conditions explicit structure provides measurable benefits. Third, it remains unclear at which level of abstraction such structure should be imposed, whether at the level of parameters, functions, or latent factors. Fourth, adoption is limited by both theoretical and practical barriers, including identifiability, generalization, and computational feasibility. Finally, the community must address the tension between building models that are interpretable from the start and those that rely on post-hoc explanations. The following subsections provide a more detailed discussion of these five questions.

#### Can Reusable Modules Enable Portability Across Tasks?
A central question is whether the skills or routines acquired by large models can be isolated and reused as portable modules across tasks without reducing overall performance [@doi:10.48550/arXiv.2108.07258]. The vision of modularity is to build an ecosystem of specialized components that can be composed when needed, instead of training a new large model for each task. Promising approaches operate at different levels: (i) representation-level constraints such as concept bottlenecks enforcing human-understandable features; (ii) memory-based mechanisms such as prototype libraries for case retrieval; and (iii) architecture-level designs such as sparse adapters or routing networks that activate context-relevant modules [@doi:10.48550/arXiv.2404.13628; @doi:10.48550/arXiv.2311.16142].

Applications illustrate the promise of this research. In healthcare, diagnostic modules could be reused across diseases. In natural language processing, syntax-aware modules might be applied across languages. However, modularity also introduces risks: interactions between modules may cause interference or instability in generalization, and poorly aligned components may propagate or amplify existing biases. Future work should therefore design evaluation protocols that test not only portability and composability, but also isolation of unintended side effects and robustness to distribution shift [@doi:10.48550/arXiv.2407.21783].

#### What Are the Theoretical and Practical Benefits of Explicit Structure?
Clarifying the theoretical and practical benefits of explicit structure is an important open question. Implicit adaptation is highly flexible, but explicit structure may provide stronger guarantees of robustness and generalization under distribution shift. Practical benefits include greater interpretability, improved debugging, and the ability to incorporate domain knowledge directly.

To advance this agenda, systematic comparisons with implicit approaches are needed. Stress testing under covariate shift, concept drift, long-tail distributions, and adversarial correlations is particularly important, and benchmarks such as WILDS provide a useful starting point [@doi:10.48550/arXiv.2012.07421]. At the same time, researchers must weigh the costs of explicit structure. These costs include additional annotation, increased hyperparameter complexity, and potential reductions in in-domain accuracy [@doi:10.48550/arXiv.2004.07780, @doi:10.48550/arXiv.1911.08731]. A comprehensive evaluation framework that quantifies both theoretical guarantees and practical trade-offs remains to be established.

#### At What Level of Abstraction Should Explicit Structure Be Imposed?
Determining the appropriate level of abstraction for intervention remains a challenge. Parameter-level edits provide precise control but are brittle and can have unpredictable side effects [@doi:10.48550/arXiv.2212.10559]. Concept-level interventions provide stability and interpretability but may fail to capture the model’s internal computations in full detail [@doi:10.48550/arXiv.1807.03124].

Intermediate levels may offer a balance. For example, function-level interventions or local surrogate models can capture mid-level abstractions that combine precision with stability. More importantly, future work should aim to develop methods that allow translation across levels. For instance, low-level parameter edits could be distilled into high-level conceptual summaries, while abstract concepts might be operationalized through concrete parameter changes. Such tools would make adaptive models more interpretable and more controllable in practice.

#### What Theoretical and Practical Barriers Remain?
Several barriers continue to limit the adoption of adaptive models. On the theoretical side, researchers have yet to establish strong guarantees for identifiability and generalization under distribution shift [@doi:10.48550/arXiv.1911.08731]. Extending these guarantees to high-dimensional and multimodal data remains an unsolved challenge.

Practical barriers are equally important. Training and deploying adaptive models requires significant computational and memory resources. Data limitations, such as biased sampling and noisy feedback, reduce reliability. Evaluation frameworks remain centered on accuracy, with insufficient attention to fairness, stability, and long-term robustness. Finally, the absence of standardized tools and implementation guidelines prevents many practitioners from applying state-of-the-art methods beyond research settings [@doi:10.3390/publications13020019].

#### Interpretable-by-Design vs Post-hoc Interpretability: What Is the Right Path Forward?
A final open question concerns the balance between interpretable-by-design approaches and post-hoc interpretability. Interpretable-by-design models, such as varying coefficient models, provide transparency and faithfulness from the outset but may restrict predictive performance [@doi:10.1111/j.2517-6161.1993.tb01939.x]. Post-hoc methods allow powerful foundation models to be explained after training, but explanations may be incomplete or unfaithful to the model’s internal reasoning [@doi:10.48550/arXiv.2108.07258].

Progress in both directions suggests that the future lies in integration rather than a binary choice. Hybrid models may embed interpretable structures at their core while using post-hoc tools for flexibility. Promising directions include benchmarks that jointly evaluate adaptivity and interpretability, as well as human-in-the-loop workflows that allow domain experts to constrain and validate model adaptation in practice.

### Broader Challenges and Future Outlook

Emerging paradigms such as Agentic Context Engineering (ACE) push this vision further by treating the context itself as an adaptive, evolving entity. In this framework, language models continuously refine and regenerate their own contexts through feedback, reflection, and planning, enabling self-improving adaptation cycles across time [@doi:10.48550/arXiv.2510.04618]. While the previous section focused on research questions that can be addressed by new methods, theory, and experiments, broader challenges remain that extend beyond purely technical considerations. These challenges concern the responsible deployment of adaptive models in real-world environments, where issues such as ethics, fairness, and regulatory compliance play a critical role. Adaptive systems used in sensitive domains such as healthcare and finance must satisfy principles of interpretability, auditability, and accountability to prevent harm and maintain public trust [@doi:10.48550/arXiv.2108.07258]. Collaboration between regulators, practitioners, and researchers is essential to establish transparent auditing standards and verifiable documentation for adaptive decisions.

Another set of challenges arises from the dynamic interaction between adaptive models and their environments. Feedback loops may amplify small initial biases, leading to systematic disadvantages for certain groups over time. Examples can be seen in credit scoring, hiring, and online recommendation systems, where early decisions influence future data collection and can entrench inequalities [@doi:10.1145/3097983.3098066]. Addressing these risks requires methods that anticipate long-term effects, including simulation studies, formal analyses of dynamic systems, and model designs that incorporate fairness constraints directly during learning.

Looking ahead, the long-term vision for adaptive modeling is to develop systems that are not only powerful but also trustworthy. Progress requires moving beyond accuracy as the dominant evaluation criterion to include fairness, stability, and transparency. Human oversight should be an integral part of adaptive pipelines, enabling experts to guide and validate model behavior in practice. Sustainability is another important dimension, as the computational and environmental costs of adaptive models continue to grow. By combining technical innovation with responsible deployment, the field can ensure that adaptive inference contributes to both scientific progress and societal benefit.


## Conclusion

### Overview of Insights
This review established a unifying framework for understanding context-adaptive inference across both explicit statistical models and implicit adaptation in modern foundation models. By tracing how adaptation appears in parameterized functions such as varying-coefficient models and in emergent processes like in-context learning, we showed that these paradigms share a common estimator form and theoretical foundation.

Across the literature, a consistent pattern emerges: adaptivity becomes effective when context, computation, and interpretation are aligned. The principles of context-aware efficiency integrate these aspects, clarifying when adaptation enhances robustness and when it introduces instability. Within this perspective, model design choices can be connected to measurable outcomes such as data efficiency, modularity, and transferability, grounding the abstract notion of adaptivity in verifiable performance.

The unified view presented in this review connects statistical inference with ideas from machine learning and cognitive modeling, where adaptive reasoning and context-sensitive generalization are regarded as key components of intelligent behavior. Cognitive theories have long emphasized that efficient adaptation arises from internal models that balance precision and flexibility, an idea now mirrored in recent computational analyses of in-context learning [@doi:10.48550/arXiv.2506.17859]. By bridging these perspectives, this framework provides both a conceptual foundation and a practical guide for developing adaptive systems that are interpretable, reliable, and scalable.

#### Context-Aware Efficiency: A Unifying Framework

The principles of context-aware efficiency emerge as a unifying theme across the diverse methods surveyed in this review. This framework provides a systematic approach to designing methods that are both computationally tractable and statistically principled.

Several fundamental insights emerge from our analysis. Rather than being a nuisance parameter, context provides information that can be leveraged to improve both statistical and computational efficiency. Methods that adapt their computational strategy based on context often achieve better performance than those that use fixed approaches. The design of context-aware methods requires careful consideration of how to balance computational efficiency with interpretability and regulatory compliance.

Recent studies also demonstrate that context-adaptive strategies can emerge spontaneously in large models trained on diverse tasks, linking computational efficiency to rational inference principles [@doi:10.48550/arXiv.2507.16003]. These findings suggest that implicit adaptation can serve as a computational analog of Bayesian updating, where context dynamically reweights prior knowledge to improve generalization. Similar ideas have been explored in meta-learning frameworks such as MetaICL, which meta-trains language models to acquire reusable adaptation strategies through exposure to varied task distributions [@doi:10.48550/arXiv.2110.15943].

Future research in context-aware efficiency should focus on developing methods that can efficiently handle high-dimensional, multimodal context information, creating systems that can adaptively allocate computational resources based on context complexity and urgency, investigating how efficiency principles learned in one domain can be transferred to others, and ensuring that context-aware efficiency methods can be deployed in regulated environments while maintaining interpretability [@doi:10.48550/arXiv.2510.04618].

The development of context-aware efficiency principles has implications beyond statistical modeling. More efficient methods reduce computational costs and environmental impact, enabling sustainable computing practices. Efficient methods also democratize AI by enabling deployment of sophisticated models on resource-constrained devices. Furthermore, context-aware efficiency enables deployment of personalized models in time-critical applications, supporting real-time decision making.

As we move toward an era of increasingly personalized and context-aware statistical inference, the principles outlined in this review provide a foundation for developing methods that are both theoretically sound and practically useful.


### Future Directions
Looking ahead, the evolution of context-adaptive inference will likely proceed along four interconnected paths.

#### Theoretical Foundations
Future research should formalize implicit adaptation within a consistent statistical framework, linking neural computation to principles of efficiency, identifiability, and invariance. Clarifying these theoretical connections will support better understanding of when implicit adaptation approximates explicit statistical reasoning and how both approaches can be integrated. Recent advances have begun to view in-context learning as an emergent form of structure induction, suggesting that large models implicitly learn compositional representations that approximate rational inference processes [@doi:10.48550/arXiv.2506.17859].

#### Modular and Compositional Methods
Progress in parameter-efficient fine-tuning, compositional adaptation, and reusable modules will make large models more flexible and controllable. Building libraries of specialized components that can be dynamically combined will promote efficient reuse and domain transfer while maintaining interpretability. Work on tabular in-context learning, such as the TabICL architecture, illustrates how these principles can scale to structured data domains while preserving modular control and generalization [@doi:10.48550/arXiv.2502.05564].

#### Evaluation and Reliability
Developing standardized benchmarks that jointly assess robustness, calibration, and interpretability is essential for advancing both theory and application. Future evaluation frameworks should emphasize context-stratified performance, long-term stability, and transparent reporting of adaptation behavior under distribution shifts. Ongoing analyses of the stability and transience of in-context strategies [@doi:10.48550/arXiv.2507.16003] underscore the importance of evaluating not only short-term generalization but also the persistence and reproducibility of adaptive behavior across training regimes.

#### Responsible and Sustainable Deployment
As adaptive systems become embedded in decision-making processes, integrating fairness auditing, human oversight, and energy efficiency into their design will be critical for ensuring public trust. Addressing the environmental cost of large-scale adaptation and developing resource-conscious algorithms will also contribute to sustainable computing practices. Emerging work on efficient foundation models and rational adaptation frameworks [@doi:10.48550/arXiv.2510.04618] highlights how technical design and ethical responsibility can be jointly optimized in real-world deployment.

Together, these directions outline a path toward the next generation of adaptive models that are both powerful and trustworthy. Progress will depend on combining rigorous statistical understanding with transparent design and responsible deployment, moving steadily toward the broader goal of making implicit adaptation explicit and accountable.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>


## Appendix A {.page_break_before}

This appendix gives full proofs for Proposition 1 and Corollary 1. We keep the weighted support-set notation from the Introduction and make all linear-algebra steps explicit.

---

### A.0  Preliminaries and identities

- **Joint features.** For any pair $(x,c)$, define  
  $$
  \psi(x,c):=x\otimes \phi(c)\in\mathbb{R}^{d_x d_c}.
  $$
  For each indexed training example $a$ (standing in for $(i,j)$), write $\psi_a:=\psi(x_a,c_a)$.

- **Design/labels/weights.** Stack $N=\sum_i m_i$ training rows:
  $$
  Z\in\mathbb{R}^{N\times d_x d_c}\ \text{ with rows } Z_a=\psi_a^T,\qquad
  y\in\mathbb{R}^{N},\qquad
  W=\mathrm{diag}(w_a)\in\mathbb{R}^{N\times N},\ w_a\ge 0.
  $$
  Define the (unweighted) Gram matrix $K:=ZZ^\top$ and the weighted Gram
  $$
  K_W:=W^{1/2} K\, W^{1/2} \;=\; W^{1/2} Z Z^\top W^{1/2}.
  $$
  For a query $(x,c)$, let $k(\cdot,(x,c)) := Z\,\psi(x,c)\in\mathbb{R}^N$ and $k_{(x,c)}:=W^{1/2}k(\cdot,(x,c))$.

- **Vectorization identity.** For conformable matrices $A,B,C$,
  $$
  \mathrm{vec}(A B C)=\big(C^\top\otimes A\big)\mathrm{vec}(B),\quad
  \langle \mathrm{vec}(B),\,x\otimes z\rangle = x^\top B z.
  $$

- **Weighted ridge solution.** For any $X\in\mathbb{R}^{N\times p}$, ridge objective
  $$
  \min_\beta \ \|W^{1/2}(y-X\beta)\|_2^2+\lambda\|\beta\|_2^2
  $$
  has unique minimizer $\widehat\beta=(X^\top W X+\lambda I)^{-1}X^\top W y$ and equivalent dual form
  $$
  \widehat\beta = X^\top W^{1/2}\big(W^{1/2}XX^\top W^{1/2}+\lambda I\big)^{-1}W^{1/2}y.
  $$
  Predictions for a new feature vector $x_\star$ equal
  $$
  \widehat f(x_\star)=x_\star^\top \widehat\beta
  \;=\;
  \underbrace{\big(W^{1/2}X x_\star\big)^\top}_{k_\star^\top}
  \big(W^{1/2}XX^\top W^{1/2}+\lambda I\big)^{-1}
  W^{1/2}y.
  $$
  This is **kernel ridge regression** (KRR) with kernel $K_W=W^{1/2}XX^\top W^{1/2}$ and query vector $k_\star=W^{1/2}X x_\star$.


---

### A.1  Proof of Proposition 1(A): explicit varying-coefficients ⇔ weighted KRR on joint features

Assume the linear, squared-loss setting with $y=\langle \theta(c),x\rangle+\varepsilon$ and $\mathbb{E}[\varepsilon]=0$.  
Let the varying-coefficients model be $\theta(c)=B\,\phi(c)$ with $B\in\mathbb{R}^{d_x\times d_c}$ and ridge penalty $\lambda\|B\|_F^2$.

**Step 1 (reduce to ridge in joint-feature space).**  
Vectorize $B$ as $\beta=\mathrm{vec}(B)\in\mathbb{R}^{d_x d_c}$.  
By the identity above,
$$
x_a^T B\,\phi(c_a)
= \langle \beta,\, x_a\otimes \phi(c_a)\rangle
= \langle \beta,\,\psi_a\rangle.
$$
Thus the weighted objective specialized from (★) is
$$
\min_{\beta\in\mathbb{R}^{d_x d_c}}
\ \big\|W^{1/2}\big(y - Z\beta\big)\big\|_2^2 + \lambda \|\beta\|_2^2,
$$
which is exactly weighted ridge with design $X\equiv Z$.

**Step 2 (closed form and prediction).**  
By the ridge solution,
$$
\widehat\beta=(Z^T W Z+\lambda I)^{-1} Z^T W y,
$$
and the prediction at a query $(x,c)$ with joint feature $\psi(x,c)$ is
$$
\widehat y(x,c)=\psi(x,c)^T \widehat\beta
= \underbrace{\big(W^{1/2} Z\, \psi(x,c)\big)}_{k_{(x,c)}}^\top
\big(W^{1/2} Z Z^\top W^{1/2}+\lambda I\big)^{-1} W^{1/2} y.
$$

**Step 3 (kernel form).**  
Since $K:=ZZ^T$ and $K_W:=W^{1/2} K W^{1/2}$,
$$
\boxed{\ \widehat y(x,c)\;=\; k_{(x,c)}^T \big(K_W+\lambda I\big)^{-1} W^{1/2}y\ }.
$$
Moreover, the $(a,b)$-th entry of the kernel matrix $K$ is 
$$
K_{ab}=\langle \psi_a,\psi_b\rangle
=\big\langle x_a\otimes \phi(c_a),\,x_b\otimes \phi(c_b)\big\rangle
=\langle x_a,x_b\rangle\cdot \langle \phi(c_a),\phi(c_b)\rangle,
$$
so (A) is precisely **KRR on joint features** with sample weights $W$.  
This proves part (A). ■


---

### A.2  Proof of Proposition 1(B): linear ICL ⇒ kernel regression

We analyze a single attention layer operating on the weighted support set $S(c)$, using **linear** maps for queries, keys, and values:
$$
q(x,c)=Q\,\psi(x,c),\qquad k_a = K\,\psi_a,\qquad v_a=V\,\psi_a,
$$
with $Q\in\mathbb{R}^{d_q\times d_\psi}$, $K\in\mathbb{R}^{d_k\times d_\psi}$, $V\in\mathbb{R}^{d_v\times d_\psi}$, $d_\psi=d_x d_c$. Let the **unnormalized** attention score for index $a$ be
$$
s_a(x,c):=w_a\,\langle q(x,c),k_a\rangle \;=\; w_a\,\psi(x,c)^T Q^T K\,\psi_a .
$$
Define normalized weights $\alpha_a(x,c):=s_a(x,c)/\sum_b s_b(x,c)$ (or any fixed positive normalization; the form below is pointwise in $\{\alpha_a\}$). The context representation and scalar prediction are
$$
z(x,c)=\sum_a \alpha_a(x,c)\, v_a,\qquad \widehat y(x,c)=u^T z(x,c).
$$

We prove two statements: **(B1)** exact KRR if the attention maps are fixed and only the readout is trained, and **(B2)** kernel regression with the NTK if the attention parameters are trained in the linearized regime.

#### A.2.1  (B1) Fixed attention, trained linear head ⇒ exact KRR

Assume $Q,K,V$ are fixed functions (pretrained or chosen a priori), hence $\alpha_a(x,c)$ are **deterministic** functions of $(x,c)$ and the support set. Define the induced **feature map**
$$
\varphi(x,c):=\sum_a \alpha_a(x,c)\, v_a \;\in\; \mathbb{R}^{d_v}.
$$
Stack $\varphi_a:=\varphi(x_a,c_a)$ row-wise into $\Phi\in\mathbb{R}^{N\times d_v}$. Training only the readout $u$ with weighted ridge,
$$
\widehat u \in \arg\min_u \ \|W^{1/2}(y-\Phi u)\|_2^2+\lambda \|u\|_2^2
$$
yields $\widehat u=(\Phi^T W \Phi + \lambda I)^{-1}\Phi^T W y$ and predictions
$$
\widehat y(x,c)=\varphi(x,c)^T \widehat u
= \underbrace{\big(W^{1/2}\Phi\,\varphi(x,c)\big)}_{k_{(x,c)}}^T
\big(W^{1/2}\Phi\Phi^T W^{1/2}+\lambda I\big)^{-1} W^{1/2} y.
$$
Therefore,
$$
\boxed{\ \widehat y(x,c)=k_{(x,c)}^T \big(K_W+\lambda I\big)^{-1}W^{1/2}y\ },
\quad K_W:=W^{1/2}\underbrace{(\Phi\Phi^T)}_{=:K}W^{1/2},
$$
which is exactly **kernel ridge regression** with kernel
$$
k\big((x,c),(x',c')\big)=\langle \varphi(x,c),\varphi(x',c')\rangle.
$$
Because $v_a=V\psi_a$ and $\alpha_a(x,c)\propto w_a\,\psi(x,c)^T Q^T K \psi_a$, $\varphi$ is a linear transform of a **weighted average of joint features**; hence the kernel is a dot-product on linear transforms of $\{\psi_a\}$. This proves (B1). ■


#### A.2.2  (B2) Training attention in the linearized/NTK regime ⇒ kernel regression with NTK

Now let $\theta=(Q,K,V,u)$ be trainable, and suppose training uses squared loss with gradient flow (or sufficiently small steps) starting from initialization $\theta_0$. The **linearized model** around $\theta_0$ is the first-order Taylor expansion
$$
\widehat y_\theta(x,c)\;\approx\;\widehat y_{\theta_0}(x,c)+\nabla_\theta \widehat y_{\theta_0}(x,c)^T (\theta-\theta_0)
=: \widehat y_{\theta_0}(x,c) + \phi_{\mathrm{NTK}}(x,c)^T (\theta-\theta_0),
$$
where $\phi_{\mathrm{NTK}}(x,c):=\nabla_\theta \widehat y_{\theta_0}(x,c)$ are the **tangent features**. Standard NTK results (for squared loss, gradient flow, and linearization-validity conditions) imply that the learned function equals **kernel regression with the NTK**:
$$
k_{\mathrm{NTK}}\big((x,c),(x',c')\big)
:= \big\langle \phi_{\mathrm{NTK}}(x,c),\,\phi_{\mathrm{NTK}}(x',c')\big\rangle,
$$
i.e., predictions have the KRR form with kernel $K_{\mathrm{NTK}}$ on the training set (and explicit ridge if used, or implicit regularization via early stopping).

It remains to identify the structure of $\phi_{\mathrm{NTK}}$ for our **linear attention** block and show it lies in the span of **linear transforms of joint features**. Differentiating
$\widehat y(x,c)=u^T \sum_a \alpha_a(x,c)\, V\psi_a$ at $\theta_0$ yields four groups of terms:

- **Readout path ($u$).** $\partial \widehat y/\partial u = \sum_a \alpha_a(x,c)\, V\psi_a = \varphi_0(x,c)$. This is linear in $\{\psi_a\}$.

- **Value path ($V$).** $\partial \widehat y/\partial V = \sum_a \alpha_a(x,c)\, u\,\psi_a^T$. This contributes terms of the form $(u\otimes I)\sum_a \alpha_a(x,c)\psi_a$, i.e., linear in $\{\psi_a\}$.

- **Query/key paths ($Q,K$).** For linear attention with scores $s_a=w_a\,\psi(x,c)^T Q^T K \psi_a$ and normalized $\alpha_a=s_a/\sum_b s_b$, derivatives of $\alpha_a$ w.r.t. $Q$ and $K$ are linear combinations of $\psi(x,c)$ and $\{\psi_a\}$:
  $$
  \frac{\partial \alpha_a}{\partial Q}\propto 
  \sum_b \big[\delta_{ab}-\alpha_b(x,c)\big]\,
  w_a w_b \big( K\psi_a\,\psi(x,c)^T \big),
  \qquad
  \frac{\partial \alpha_a}{\partial K}\propto 
  \sum_b \big[\delta_{ab}-\alpha_b(x,c)\big]\,
  w_a w_b \big( \psi(x,c)\,\psi_a^T Q^T \big),
  $$
  and hence $\partial \widehat y/\partial Q$, $\partial \widehat y/\partial K$ are finite linear combinations of tensors each bilinear in $\psi(x,c)$ and some $\psi_a$. Contracting with $u$ and $V$ produces terms *linear* in $\psi(x,c)$ and linear in the set $\{\psi_a\}$.

Collecting all components, the tangent feature map can be written as
$$
\phi_{\mathrm{NTK}}(x,c)=\mathcal{L}\big(\psi(x,c),\{\psi_a\}\big),
$$
where $\mathcal{L}$ is a fixed linear operator determined by $\theta_0$, $W$, and the normalization rule for attention. Consequently, the NTK takes the **dot-product** form
$$
k_{\mathrm{NTK}}\big((x,c),(x',c')\big)=
\Psi(x,c)^T\, \mathcal{M}\, \Psi(x',c'),
$$
for some positive semidefinite matrix $\mathcal{M}$ and a finite-dimensional feature stack $\Psi$ that concatenates linear transforms of $\psi(x,c)$ and of the support-set $\{\psi_a\}$. In particular, $k_{\mathrm{NTK}}$ is a dot-product kernel on **linear transforms of the joint features** (possibly augmented by normalization-dependent combinations). Therefore, training the linear-attention ICL model in the linearized regime equals kernel regression with such a kernel—completing (B2). ■

**Assumptions for A.2.2.** Squared loss; gradient flow (or sufficiently small steps); initialization independent of the data; and a regime where the linearization error stays controlled over training (e.g., small learning rate, sufficient width/depth so that the NTK remains close to its initialization).

---

### A.3  Proof of Corollary 1: retrieval/gating/weighting as kernel/measure choices

In both A.1 and A.2, predictions have the KRR form
$$
\widehat y(x,c)=k_{(x,c)}^T \big(K^\sharp + \lambda I\big)^{-1} \mu,
$$
where $K^{\sharp}$ is a positive semidefinite kernel matrix computed over the support set (e.g., $K_W=W^{1/2}ZZ^T W^{1/2}$ in A.1 or $W^{1/2}\Phi\Phi^T W^{1/2}$ / $K_{\mathrm{NTK}}$ in A.2), $k_{(x,c)}$ is the associated query vector, and $\mu=W^{1/2}y$ (or an equivalent reweighting).

- **Retrieval $R(c)$ / gating.** Changing the support set $S(c)$ (e.g., via a retriever or a gating policy) **removes or adds rows/columns** in $K^\sharp$ and entries in $k_{(x,c)}$. This is equivalent to changing the **empirical measure** over which the kernel smoother is computed (i.e., which samples contribute and how).

- **Weights $w_{ij}(c)$.** Changing the weights modifies $W$ and hence replaces $K$ by $K_W=W^{1/2} K W^{1/2}$ and $k$ by $k_{(x,c)}=W^{1/2}k$. This is standard **importance weighting** in kernel regression.

- **Induced kernels.** Attention, value projections, or learned encoders change the **feature map** (e.g., $\psi\mapsto V\psi$ or $\psi\mapsto Q\psi$), thereby changing the kernel $k((x,c),(x',c'))=\langle \Phi(x,c),\Phi(x',c')\rangle$.

Thus retrieval/gating instantiate **neighborhood selection** (measure choice), and value/query/key processing instantiate **kernel choice**. ■

---

### A.4  Remarks

- **No Gaussianity is required.** Part (A) only uses squared loss and linear algebra; the noise model $y=f(x,c)+\varepsilon$ with $\mathbb{E}[\varepsilon]=0$ suffices.
- **Early stopping vs. explicit ridge.** If training uses early stopping rather than explicit $\lambda$, the resulting predictor is still a kernel regressor with an *implicit* regularization parameter controlled by stopping time (for gradient flow on squared loss).
- **Multiple layers / nonlinear value stacks.** With deeper nonlinear stacks, the exact identities above become local/first-order (linearized) approximations; the NTK statement continues to apply under its usual conditions.

