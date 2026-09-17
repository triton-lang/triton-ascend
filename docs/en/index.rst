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
   <a href="https://deepwiki.com/triton-lang/triton-ascend"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
   </p>

**Triton-Ascend** is an optimized version of Triton adapted for Huawei Ascend processors, providing efficient kernel auto-tuning, operator compilation, and deployment capabilities. It supports Ascend Atlas A2/A3/950 series products, maintains compatibility with Triton core syntax, and includes deep optimizations for Ascend NPU features such as automatic kernel argument parsing, memory access optimization, and robust deployment mechanisms.

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
