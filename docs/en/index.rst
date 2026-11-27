Triton Ascend
=============

.. raw:: html

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/triton-lang/triton-ascend" data-show-count="true" data-size="large" aria-label="Star triton-lang/triton-ascend on GitHub">Star</a>
   <a class="github-button" href="https://github.com/triton-lang/triton-ascend/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch triton-lang/triton-ascend on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/triton-lang/triton-ascend/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork triton-lang/triton-ascend on GitHub">Fork</a>
   </p>

   <p style="text-align:center; margin-top: 4px;">
   <a href="https://deepwiki.com/triton-lang/triton-ascend"><img src="https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="Ask DeepWiki"></a>
   </p>

**Triton-Ascend** is an optimized version of Triton adapted for Huawei Ascend processors, providing efficient kernel auto-tuning, operator compilation, and deployment capabilities. It supports Atlas A2/A3 and Ascend 950PR&950DT series products, maintains compatibility with Triton core syntax, and includes deep optimizations for Huawei Ascend NPU features such as automatic kernel argument parsing, memory access optimization, and robust deployment mechanisms.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Getting Started

   Release Notes <release_note>
   Quick Start <quick_start>
   Installation Guide <installation_guide>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Tutorials & Examples

   Vector Operator <programming_guide/vector_operator>
   Cube Operator <programming_guide/cube_operator>
   CV Fusion Operator <programming_guide/cv_fusion_operator>
   Triton-Ascend Autotune <autotune_guide>
   Triton-Ascend AOT Compile <aot_guide>
   Example Operators <examples/index>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Development Guide

   Triton-Ascend Operator Programming <programming_guide/index>
   Triton-Ascend Operator Migration <migration_guide/index>
   Triton-Ascend Operator Debugging and Profiling <debug_guide/index>
   Environment Variables and Compiler Options <environment_variable_and_compiler_options_reference>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API Reference

   triton <python-api/triton>
   triton.language <python-api/triton.language>
   triton.testing <python-api/triton.testing>
   triton.language.extra.cann.libdevice <python-api/triton.language.extra.cann.libdevice>

.. Temporarily disabled -- restore into the toctree above when needed:
   triton.language.extra.cann.extension <python-api/triton.language.extra.cann.extension>
   triton.extension.buffer.language <python-api/triton.extension.buffer.language>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Features

   Architecture Design and Core Features <architecture_design_and_core_features>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: FAQ

   Triton-Ascend FAQ <FAQ>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Community

   Contributing Guide <community/CONTRIBUTING>
   Code of Conduct <community/CODE_OF_CONDUCT>
   Governance <community/GOVERNANCE>
   Community Technical Meeting <community/community_technical_meeting>
   Maintainers <community/MAINTAINERS>
   Contributors <community/CONTRIBUTOR>
   Security Note <community/SECURITYNOTE>
