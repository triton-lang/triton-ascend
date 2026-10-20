# Contributing Guide

## Getting Started

- Fork the Triton-Ascend repository on [GitHub](https://github.com/triton-lang/triton-ascend).
- Read [README.md](https://github.com/triton-lang/triton-ascend/blob/main/README.md) file to obtain the project information and build the development environment.

## Developer Certificate of Origin (DCO)

All commits must include a `Signed-off-by:` line. Use `git commit -s` to add it automatically:

```bash
git commit -s -m "your commit message"
```

This will automatically add a line `Signed-off-by: Your Name <your.email@example.com>` at the end of your commit message, indicating that you certify the origin and authorization of the contribution.

## Developer Guide

### Coding Style

Follow the coding style below to make Triton Ascend easy to develop, maintain, and review.

- Coding Guide

  Use the unified coding style of the Triton Ascend community. The recommended coding style for Python is [PEP 8 Coding Style](https://pep8.org/), and that for C++ is [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy), [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), [CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [ShellCheck](https://github.com/koalaman/shellcheck), and [pylint](https://pylint.org/) to check the code format. You are advised to install these plug-ins in your IDE.

- Unit Test Guide

  Use the unified unit test style of the Triton Ascend community. The recommended unit test style for Python is [pytest](http://www.pytest.org/en/latest/), and that for C++ is [GoogleTest Primer](https://github.com/google/googletest/blob/main/docs/primer.md). The design intent of a test case should be reflected by its annotation name. For details about how to design Python test cases, see the [gather test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_gather.py) and [layer_norm test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/05-layer-norm.py); for compiler/MLIR conversion (C++ backend) test cases, see the examples under the [lit MLIR test directory](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/unittest/Conversion).

- Refactoring Guide

  We encourage developers to refactor our code to eliminate code smells. The refactored code should also comply with the coding style and testing style requirements. When receiving a warning, refactor the code to be merged.

### Fork-Pull Mode

1. Fork the Triton Ascend project.

    Before committing your code to the Triton Ascend project, ensure that you have forked the Triton Ascend project to your own repository. You will then develop the project in your own forked repository and merge it to the Triton Ascend project through a pull request (PR). This means that there is parallel development between the Triton Ascend repository and your own repository. Be careful to avoid inconsistency between repositories.

2. Clone the remote repository.

   Use git to clone the Triton Ascend project you have forked and add the upstream repository.

   ```shell
   git clone https://github.com/{your_forked_repo}/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
   git remote add upstream https://github.com/triton-lang/triton-ascend.git
   ```

3. Develop code locally.

   Before developing your code, you need to set up the development environment according to the [Triton Ascend Installation Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md).

   To avoid inconsistency between branches, create a new local development branch for new features.

   ```shell
   git checkout -b {new_branch_name} origin/main
   git fetch upstream       # Fetch the latest code from the upstream repository
   git rebase upstream/main # Rebase onto the latest upstream trunk
   ```

   Taking the main branch as an example, Triton Ascend may create release branches or downstream development branches as needed. Once you have created your branch and synchronized it with the latest upstream main branch, you can start developing your code.

4. Perform a self-test.

   After the code is modified, check whether the changes can pass the test.

   Write a test script for the developed code in the **ascend/examples/pytest_ut** directory of your local code branch, and verify the test script in the local environment to ensure that the changes can pass the test.

5. Push code to the remote repository.

   After updating and testing the code, push your commit to the remote repository.

   ```shell
   git add .
   git status # Check the updated files
   git commit -s -m "your commit message"
   git push origin {your_new_branch_name}
   ```

6. Create a pull request to the Triton Ascend main repository.

   After pushing code to your remote repository, create a pull request between your new branch and the Triton Ascend main branch. After the merge request is created, Jenkins CI will be automatically set to build your pipeline test. You are advised to merge your pull request to the upstream main branch as soon as possible to reduce the merge risk.

### Troubleshooting Gated Commit

Gated commit may encounter the following exceptions. Rectify the exceptions according to the related information.

- Compilation failed

  Check the cause of the compilation failure as prompted, and then recompile the code.

- Static check failed

  Find and fix the exception information in the code as prompted.

- CI pipeline failed

  Find the failed test cases of the CI pipeline as prompted, and then check the cause. After the fault is rectified, run the CI pipeline again.

### Issue Specifications

A good way to contribute to the project is to send a detailed report when you encounter a problem. We are always very grateful for detailed and thorough bug reports, and we will be very grateful to you for that!

Please include the following information when you file an issue:

- What is the software version (Triton Ascend, Python, OS, etc.) used in your environment?
- Is it a bug report or a feature request?
- What kind of issue are you reporting? Add the corresponding tag to highlight it on the issue dashboard.
- What happened?
- What did you expect to happen?
- How to reproduce the issue? (As accurately as possible)

You can choose from one of the pre-defined templates when [submitting issues of different categories](https://github.com/triton-lang/triton-ascend/issues/new/choose).

Notes for contributors:

- If you find an unresolved issue that is exactly what you are trying to solve, comment on the issue and tell others that you will be responsible for handling it.
- If the issue has existed for a period of time, it is recommended that you perform a pre-check before solving the issue.
- If you have resolved the issue you report, inform others before closing the issue.

### Pull Request Proposal

- Propose your ideas as issues on [GitHub](https://github.com/triton-lang/triton-ascend).
- If the new feature to be developed requires a large number of design details, you should also commit the design solution.
- After reaching a consensus in the issue discussion and design solution review, fork the project and commit a PR.
- No PR is allowed until you receive 2+LGTM (Looks Good To Me) from the approver. Note that you are not allowed to add LGTM to your own PRs.
- After the PR is fully discussed, it will be merged, rejected, or abandoned based on the discussion result.

## Notes

- Avoid any irrelevant changes.
- Ensure that your commit history is concise and orderly.
- Before creating a PR, please rebase the latest code from the upstream repository.
- For a bug-fixing PR, ensure that all related issues and PRs are linked.
