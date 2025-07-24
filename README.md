# Context-Adaptive Statistical Inference: Recent Progress, Open Problems, and Opportunities for Foundation Models

[![HTML Manuscript](https://img.shields.io/badge/manuscript-HTML-blue.svg)](https://adaptinfer.github.io/context-review/)
[![PDF Manuscript](https://img.shields.io/badge/manuscript-PDF-blue.svg)](https:/adaptinfer.github.io/context-review/manuscript.pdf)
[![GitHub Actions Status](https://github.com/AdaptInfer/context-review/workflows/Manubot/badge.svg)](https://github.com/AdaptInfer/context-review/actions)


This is an open, collaborative review paper on context-adaptive statistical methods. We look at recent progress, identify open problems, and find practical opportunities for applying these methods. We are particularly excited by the opportunities for foundation models to provide context for statistical inference.


This manuscript is created automatically from the content in [content](https://github.com/AdaptInfer/context-review/tree/main/content) using Manubot. Please [contribute](CONTRIBUTING.md)! Make a PR or file an issue, and see below for more information about Manubot. Live update versions of the manuscript are available at:

+ **HTML manuscript** at https://adaptinfer.github.io/context-review/
+ **PDF manuscript** at https://adaptinfer.github.io/context-review/manuscript.pdf

---

## Why are we writing this?
As statistical modeling evolves, we are witnessing two complementary approaches to integrating context.
Traditional statistical models are being expanded to allow explicit parameter adjustments based on context, making their
adaptations transparent and interpretable.
Meanwhile, large foundation models are being built that how to implicitly adapt to context, enabling impressive
performance in a wide range of tasks including in-context learning.
This review seeks to unite these two perspectives, combining the explicit adaptability of statistical models with the
powerful, implicit adjustments of foundation models.
By bringing these approaches together, we aim to provide a comprehensive overview of current progress, challenges,
and opportunities in context-adaptive inference.


### Key perspectives driving this review:

- **Complex models contain multitudes of simpler, context-specific models** – Every complex model can be understood as a combination of many smaller models, each tailored to specific contexts.
- **Explicit vs. implicit adaptation** – Statistical models explicitly adjust parameters based on context, while foundation models implicitly adapt to context. Combining these approaches offers new opportunities for robust and adaptive inference.
- **Context reshapes inference** – Context-awareness enhances both statistical and foundation models, improving personalization and accuracy in predictions.
- **In-context learning as a model of implicit adaptation** – Foundation models excel at tasks like in-context learning, showing how implicit adaptation can inform broader context-adaptive modeling efforts.
- **Foundation models as context providers** – These models offer flexible, scalable ways to incorporate context, enhancing traditional methods with richer context integration.
- **Challenging traditional assumptions** – Context-adaptive methods move beyond the assumption of homogeneity in data, enabling models to handle heterogeneous datasets more effectively.
- **Personalization through adaptation** – Uniting explicit and implicit context-adaptive models provides a path to more nuanced, personalized predictions that reflect the complexities of real-world data.

![./content/images/context_philosophies.png](./content/images/context_philosophies.png)

## Table of Contents
1. [Abstract](./content/01.abstract.md)
2. [Introduction](./content/02.introduction.md)
3. [From Population Assumptions to Context-Adaptive Inference](./content/03.overview.md)
4. [Principles of Context-Adaptive Inference](./content/04.principles.md)
5. [Explicit Adaptivity: Structured Estimation of $f(c)$](content/05.explicit.md)
6. [Implicit Adaptivity: Emergent Contextualization within Complex Models](content/06.implicit.md)
7. [Making Implicit Adaptivity Explicit: Local Models, Surrogates and Post Hoc Approximations](content/07.interpretations.md)
8. [Context-Invariant Training: A View from the Converse](content/08.invariant.md)
9. [Applications, Case Studies, and Software Tools](content/09.applications_tools.md)
10. [Future Trends and Opportunities with Foundation Models](content/10.future_trends.md)
11. [Open Problems](content/11.open_problems.md)
12. [Conclusions](content/12.conclusion.md)

## How can you contribute?
We welcome contributions from the community. Please see our [contribution guidelines](CONTRIBUTING.md) for more information.

---

<details>
  <summary><h2>Manubot</h2></summary>
  
<!-- usage note: do not edit this section -->

Manubot is a system for writing scholarly manuscripts via GitHub.
Manubot automates citations and references, versions manuscripts using git, and enables collaborative writing via GitHub.
An [overview manuscript](https://greenelab.github.io/meta-review/ "Open collaborative writing with Manubot") presents the benefits of collaborative writing with Manubot and its unique features.
The [rootstock repository](https://git.io/fhQH1) is a general purpose template for creating new Manubot instances, as detailed in [`SETUP.md`](SETUP.md).
See [`USAGE.md`](USAGE.md) for documentation how to write a manuscript.

Please open [an issue](https://git.io/fhQHM) for questions related to Manubot usage, bug reports, or general inquiries.

### Repository directories & files

The directories are as follows:

+ [`content`](content) contains the manuscript source, which includes markdown files as well as inputs for citations and references.
  See [`USAGE.md`](USAGE.md) for more information.
+ [`output`](output) contains the outputs (generated files) from Manubot including the resulting manuscripts.
  You should not edit these files manually, because they will get overwritten.
+ [`webpage`](webpage) is a directory meant to be rendered as a static webpage for viewing the HTML manuscript.
+ [`build`](build) contains commands and tools for building the manuscript.
+ [`ci`](ci) contains files necessary for deployment via continuous integration.

### Local execution

The easiest way to run Manubot is to use [continuous integration](#continuous-integration) to rebuild the manuscript when the content changes.
If you want to build a Manubot manuscript locally, install the [conda](https://conda.io) environment as described in [`build`](build).
Then, you can build the manuscript on POSIX systems by running the following commands from this root directory.

```sh
# Activate the manubot conda environment (assumes conda version >= 4.4)
conda activate manubot

# Build the manuscript, saving outputs to the output directory
bash build/build.sh

# At this point, the HTML & PDF outputs will have been created. The remaining
# commands are for serving the webpage to view the HTML manuscript locally.
# This is required to view local images in the HTML output.

# Configure the webpage directory
manubot webpage

# You can now open the manuscript webpage/index.html in a web browser.
# Alternatively, open a local webserver at http://localhost:8000/ with the
# following commands.
cd webpage
python -m http.server
```

Sometimes it's helpful to monitor the content directory and automatically rebuild the manuscript when a change is detected.
The following command, while running, will trigger both the `build.sh` script and `manubot webpage` command upon content changes:

```sh
bash build/autobuild.sh
```

### Continuous Integration

Whenever a pull request is opened, CI (continuous integration) will test whether the changes break the build process to generate a formatted manuscript.
The build process aims to detect common errors, such as invalid citations.
If your pull request build fails, see the CI logs for the cause of failure and revise your pull request accordingly.

When a commit to the `main` branch occurs (for example, when a pull request is merged), CI builds the manuscript and writes the results to the [`gh-pages`](https://github.com/AdaptInfer/context-review/tree/gh-pages) and [`output`](https://github.com/AdaptInfer/context-review/tree/output) branches.
The `gh-pages` branch uses [GitHub Pages](https://pages.github.com/) to host the following URLs:

+ **HTML manuscript** at https://adaptinfer.org/context-review/
+ **PDF manuscript** at https://adaptinfer.org/context-review/manuscript.pdf

For continuous integration configuration details, see [`.github/workflows/manubot.yaml`](.github/workflows/manubot.yaml).

</details>

<details>
  <summary><h2>License</h2></summary>

<!--
usage note: edit this section to change the license of your manuscript or source code changes to this repository.
We encourage users to openly license their manuscripts, which is the default as specified below.
-->

[![License: CC BY 4.0](https://img.shields.io/badge/License%20All-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![License: CC0 1.0](https://img.shields.io/badge/License%20Parts-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Except when noted otherwise, the entirety of this repository is licensed under a CC BY 4.0 License ([`LICENSE.md`](LICENSE.md)), which allows reuse with attribution.
Please attribute by linking to https://github.com/AdaptInfer/context-review.

Since CC BY is not ideal for code and data, certain repository components are also released under the CC0 1.0 public domain dedication ([`LICENSE-CC0.md`](LICENSE-CC0.md)).
All files matched by the following glob patterns are dual licensed under CC BY 4.0 and CC0 1.0:

+ `*.sh`
+ `*.py`
+ `*.yml` / `*.yaml`
+ `*.json`
+ `*.bib`
+ `*.tsv`
+ `.gitignore`

All other files are only available under CC BY 4.0, including:

+ `*.md`
+ `*.html`
+ `*.pdf`
+ `*.docx`

Please open [an issue](https://github.com/AdaptInfer/context-review/issues) for any question related to licensing.

</details>
