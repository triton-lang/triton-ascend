# Contributing Guidelines

- [Getting Started](#getting-started)
- [Developer Certificate of Origin (DCO)](#dco)
- [Development Guide](#development-guide)
  - [Code Style](#code-style)
  - [Fork-Pull Development Workflow](#fork-pull-development-workflow)
  - [CI Gate Exception Handling](#ci-gate-exception-handling)
  - [Issue Guidelines](#issue-guidelines)
  - [Submitting PRs](#submitting-prs)

<h2 id="getting-started">Getting Started</h2>

- Fork the Triton-Ascend repository on [GitHub](https://github.com/triton-lang/triton-ascend).
- Read [README.md](https://github.com/triton-lang/triton-ascend/blob/main/README.md) for project information and instructions on setting up the development environment.

<h2 id="dco">Developer Certificate of Origin (DCO)</h2>

All commits must include a `Signed-off-by:` line. Use `git commit -s` to add it automatically:

```bash
git commit -s -m "your commit message"
```

This appends a line such as `Signed-off-by: Your Name <your.email@example.com>` to the end of the commit message, indicating that you certify the origin and authorization of the contribution.

<h2 id="development-guide">Development Guide</h2>

- **[Code Style](#code-style)**
- **[Fork-Pull Development Workflow](#fork-pull-development-workflow)**
- **[CI Gate Exception Handling](#ci-gate-exception-handling)**
- **[Issue Guidelines](#issue-guidelines)**
- **[Submitting PRs](#submitting-prs)**

<h2 id="code-style">Code Style</h2>

Please follow the coding guidelines below to keep Triton-Ascend easy to develop, maintain, and review.

- Coding Guidelines

  Please use the unified coding style of the Triton-Ascend community. The recommended Python coding style is the [PEP 8 style guide](https://pep8.org/), and the recommended C++ coding style is the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy), [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), [CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [ShellCheck](https://github.com/koalaman/shellcheck), and [pylint](https://pylint.org/) to check the format of your code. It is recommended to install these plugins in your IDE.

- Unit Testing Guidelines

  Please use the unified unit testing style of the Triton-Ascend community. The recommended Python unit testing style is [pytest](http://www.pytest.org/en/latest/), and the recommended C++ unit testing style is the [Googletest Primer](https://github.com/google/googletest/blob/main/docs/primer.md). The design intent of a test case should be reflected by its comment name. For test case design, refer to the [gather test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_gather.py) and the [layer_norm test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/05-layer-norm.py).

- Refactoring Guidelines

  We encourage developers to refactor our code to eliminate "code smells". Refactored code should also follow the coding style and testing style requirements. When you receive a warning, you need to refactor the code to be merged.

<h2 id="fork-pull-development-workflow">Fork-Pull Development Workflow</h2>

1. Fork the Triton-Ascend project

   Before submitting your own code to the Triton-Ascend project, make sure you have forked the Triton-Ascend project into your own repository. You will then develop on your own forked project and merge it into the Triton-Ascend project via Pull Request. This means that there is parallel development between the Triton-Ascend repository and your own repository, so please keep the repositories consistent.

2. Clone the remote repository

   Use git to clone your forked Triton-Ascend project and add the upstream repository:

   ```shell
   git clone https://github.com/{your_forked_repo}/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
   git remote add upstream https://github.com/triton-lang/triton-ascend.git
   ```

3. Develop code in your local environment

   Before developing your code, set up the development environment according to the [Triton-Ascend Installation Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md).

   To avoid inconsistency between branches, create a new local development branch for new feature development:

   ```shell
   git checkout -b {new_branch_name} origin/main
   git fetch upstream       # Fetch the latest code from the upstream repository
   git rebase upstream/main # Rebase onto the latest upstream trunk
   ```

   Taking the main branch as an example, Triton-Ascend may create version branches or downstream development branches as needed. After creating a branch and syncing with the upstream main branch, you can start developing your code.

4. Self-test your code changes

   After completing your code changes, check whether your changes pass the tests:

   Write test cases for your code under the ascend/examples/pytest_ut path of your local code branch, and verify your test scripts in the local environment to ensure your changes pass the tests.

5. Push code to the remote repository

   After the code is updated and tested, push your commit to your remote repository.

   ```shell
   git add .
   git status #Check the updated files
   git commit -s -m "your commit message"
   git push origin {your_new_branch_name}
   ```

6. Create a Pull Request to the Triton-Ascend main repository

   After pushing the code to your remote repository, you need to create a Pull Request between your new branch and the Triton-Ascend main branch. After the merge request is created, "Jenkins CI" will automatically set up a build pipeline test for you. Please merge your Pull Request into the upstream main branch as soon as possible to reduce merge risks.

<h2 id="ci-gate-exception-handling">CI Gate Exception Handling</h2>

CI gate exceptions mainly include the following cases. Please resolve the gate exception issues according to the relevant prompt information.

- Compilation failure

  Check the cause of the compilation failure according to the prompt information, resolve it, and recompile.

- Static check failure

  Find the exception information in the code according to the prompt information and resolve it.

- CI pipeline not passed

  Find the test cases that failed in the CI pipeline according to the prompt information, check the cause, and re-run the CI pipeline after resolving it.

<h2 id="issue-guidelines">Issue Guidelines</h2>

A good way to contribute to the project is to send a detailed report when you encounter a problem. We always appreciate well-written and thorough bug reports, and we thank you for them!

When reporting an issue, please refer to the following format:

- What software versions are used in your environment (Triton-Ascend, Python, OS, etc.)?
- Is this a bug report or a feature request?
- What kind of problem are you reporting? Add the corresponding label so it stands out on the issue dashboard.
- What happened?
- What did you expect to happen?
- How to reproduce it? (As precise as possible)

You can also choose one of the predefined [issue templates](https://github.com/triton-lang/triton-ascend/issues/new/choose).

Issue consultation:

- If you find an unresolved issue that is exactly what you want to solve, please comment on the issue to tell others that you will take charge of it.
- If an issue has been open for a long time, please pre-check it before solving it.
- If you solved an issue you reported yourself, let others know before closing the issue.

<h2 id="submitting-prs">Submitting PRs</h2>

- Propose your idea as an issue on [GitHub](https://github.com/triton-lang/triton-ascend).
- If the new feature to be developed requires extensive design details, you should also submit a design proposal.
- After the issue discussion and design proposal review reach a consensus, proceed with fork development and submit a PR.
- No PR is allowed until you receive 2+ LGTM (Looks Good To Me) from Approvers. Note that approvers are not allowed to add LGTM to their own PRs.
- After the PR is fully discussed, it will be merged, rejected, or abandoned based on the discussion result.

## Notes

- Avoid any unrelated changes.
- Ensure that your commit history is concise and orderly.
- Rebase the latest code of the upstream repository before creating a PR.
- For bug fix PRs, make sure to link all related Issues and PRs.
