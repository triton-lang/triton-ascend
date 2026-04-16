# Triton Ascend Contribution Guide

- [Getting Started](#getting-started.md)
- [Developer Guide](#developer-guide.md)
  - [Coding Style](#coding-style.md)
  - [Fork-Pull Mode](#forkpull-mode.md)
  - [Troubleshooting Gated Commit](#troubleshooting-gated-commit.md)
  - [Issue Specifications](#issue-specifications.md)
  - [Pull Request Proposal](#pull-request-proposal.md)

<h2 id="getting-started.md">Getting Started</h2>

- Fork the Triton Ascend repository on [GitCode](https://gitcode.com/Ascend/triton-ascend).
- Check the [README.md](https://gitcode.com/Ascend/triton-ascend/blob/master/README.md) file to obtain the project information and build the development environment.



<h2 id="developer-guide.md">Developer Guide</h2>

- **[Coding Style](#coding-style.md)**
- **[Fork-Pull Mode](#forkpull-mode.md)**
- **[Troubleshooting Gated Commit](#troubleshooting-gated-commit.md)**
- **[Issue Specifications](#issue-specifications.md)**
- **[Pull Request Proposal](#pull-request-proposal.md)**



<h2 id="coding-style .md">Coding Style</h2>

Follow the coding style below to make Triton Ascend easy to develop, maintain, and review.

- Coding Guide

  Use the unified coding style of the Triton Ascend community. The recommended coding style for Python is [PEP 8 Coding Style](https://pep8.org/), and that for C++ is [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy), [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), [CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [ShellCheck](https://github.com/koalaman/shellcheck), and [pylint](https://pylint.org/) to check the code format. You are advised to install these plug-ins in your IDE.

- Unit Test Guide

  Use the unified unit test style of the Triton Ascend community. The recommended unit test style for Python is [pytest](http://www.pytest.org/en/latest/), and that for C++ is [GoogleTest Primer](https://github.com/google/googletest/blob/master/docs/primer.md). The design intent of a test case should be reflected by its annotation name. For details about how to design test cases, see [gather Test Cases](https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/pytest_ut/test_gather.py) and [layer_norm Test Cases](https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/tutorials/03-layer-norm.py).

- Refactoring Guide

  We encourage developers to refactor our code to eliminate code smells. The refactored code should also comply with the coding style and testing style requirements. When receiving a warning, refactor the code to be merged.



<h2 id="Fork-Pull Mode .md">Fork-Pull Mode</h2>

1. Fork the Triton Ascend project.

Before committing your code to the Triton Ascend project, ensure that you have forked the Triton Ascend project to your own repository. You will then develop the project in your own forked repository and merge it to the Triton Ascend project through a pull request (PR). This means that there is parallel development between the Triton Ascend repository and your own repository. Be careful to avoid inconsistency between repositories.

2. Clone a remote repository.

Use git to clone the Triton Ascend project you have forked and add the upstream repository.

```shell
git clone https://github.com/triton-lang/{your_forked_repo}/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
git remote add upstream https://github.com/triton-lang/triton-ascend.git
```

3. Develop code locally.

Before developing your code, you need to set up the development environment according to the [Triton Ascend Installation Guide](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md).

To avoid inconsistency between branches, create a new local development branch for new features.

```shell
git checkout -b {new_branch_name} origin/master
git fetch upstream #Fetch the latest code from the upstream repository
git rebase upstream/master #Rebase onto the latest upstream
```

Taking the master branch as an example, Triton Ascend may create version branches or downstream development branches as required. After creating a branch and synchronizing the upstream master branch, you can start developing your code.

4. Perform a self-test.

After the code is modified, check whether the changes can pass the test.

Write a test script for the developed code in the **ascend/examples/pytest_ut** directory of your local code branch, and verify the test script in the local environment to ensure that the changes can pass the test.

5. Push code to the remote repository.

After updating and testing the code, push your commit to the remote repository.

```shell
git add .
git status #Check the updated files
git commit -m "Your commit title"
git commit -s --amend #Add the concrete description of your commit
git push origin {your_new_branch_name}
```

6. Create a pull request to the Triton Ascend main repository.

After pushing code to your remote repository, create a pull request between your new branch and the Triton Ascend master branch. After the merge request is created, Jenkins CI will be automatically set to build your pipeline test. You are advised to merge your pull request to the upstream master branch as soon as possible to reduce the merge risk.

The pipeline execution process after a PR is committed is as follows:

- Comment /compile to start the pipeline test. If the test fails, modify the code as prompted and comment /compile again to trigger the pipeline test. After the test is passed, the tag ci-pipeline-passed is added.

  ```shell
  /compile
  ```

- If SC-FAIL is displayed, check the modification and comment compile#openlibing to manually trigger the check. After the check is passed, the tag SC-SUCC is added.

  ```tex
  compile#openlibing
  ```

- After the pipeline passes the test (the ci-pipeline-passed and SC-SUCC tags are added), comment @committers as prompted to review the code so that the code can be quickly merged.



<h2 id="Troubleshooting Gated Commit.md">Troubleshooting Gated Commit</h2>

Gated commit may encounter the following exceptions. Rectify the exceptions according to the related information.

- Compilation failed

  Check the cause of the compilation failure as prompted, and then recompile the code.

- Static check failed

  Find and fix the exception information in the code as prompted.

- CI pipeline failed

  Find the failed test cases of the CI pipeline as prompted, and then check the cause. After the fault is rectified, run the CI pipeline again.
  
  

<h2 id="issue-specifications.md">Issue Specifications</h2>

A good way to contribute to the project is to send a detailed report when you encounter a problem. We are always very grateful for detailed and thorough bug reports, and we will be very grateful to you for that!

Please include the following information when you file an issue:

- What is the software version (Triton Ascend, Python, OS, etc.) used in your environment?
- Is it a bug report or a functional request?
- What kind of issue are you reporting? Add the corresponding tag to highlight it on the issue dashboard.
- What's happened?
- What did you expect to happen?
- How to reproduce the issue? (As accurately as possible)

For details about the templates for filling in issues of different categories, see [Issue Specifications](https://gitcode.com/Ascend/triton-ascend/issues/create/choose).

Notes for contributors:

- If you find an unresolved issue that is exactly what you are trying to solve, comment on the issue and tell others that you will be responsible for handling it.
- If the issue has existed for a period of time, it is recommended that you perform a pre-check before solving the issue.
- If you have resolved the issue you report, inform others before closing the issue.


<h2 id="pull-request-proposal.md">Pull Request Proposal</h2>

- Propose your ideas as issues.
- If the new feature to be developed requires a large number of design details, you should also commit the design solution.
- After reaching a consensus in the issue discussion and design solution review, fork the project and commit a PR.
- No PR is allowed until you receive 2+LGTM (Looks Good To Me) from the approver. Note that you are not allowed to add LGTM to your own PRs.
- After the PR is fully discussed, it will be merged, rejected, or abandoned based on the discussion result.

### Notes:

-   Avoid any irrelevant changes.
-   Ensure that your commit history is concise and orderly.
-   Before creating a PR, please rebase the latest code from the upstream repository.
-   For a bug-fixing PR, ensure that all related issues and PRs are linked.
