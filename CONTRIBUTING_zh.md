# Triton Ascend贡献指南

- [贡献者许可协议](#贡献者许可协议.md)
- [入门](#入门.md)
- [开发指导](#开发指导.md)
  - [代码风格](#代码风格.md)
  - [Fork-Pull开发模式](#Fork-Pull开发模式.md)
  - [代码门禁异常处理](#代码门禁异常处理.md)
  - [ISSUE规范](#ISSUE规范.md)
  - [提出PR](#提出PR.md)

<h2 id="贡献者许可协议.md">
    贡献者许可协议
</h2>

在您第一次向Triton Ascend社区提交代码之前，需要签署CLA。

对于个人贡献者，签署CLA详细信息参考 [cla使用指南](https://gitcode.com/Ascend/infrastructure/blob/master/docs/cla/cla使用指南.md#faq)

CLA签署地址 [sign](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)



<h2 id="入门.md">入门</h2>

- 在[GitCode](https://gitcode.com/Ascend/triton-ascend)上Fork Triton Ascend存储库。
- 阅读[README.md](https://gitcode.com/Ascend/triton-ascend/blob/master/README.md)获取项目信息和构建开发环境。



<h2 id="开发指导.md">开发指导</h2>

- **[代码风格](#代码风格.md)**
- **[Fork-Pull开发模式](#Fork-Pull开发模式.md)**
- **[代码门禁异常处理](#代码门禁异常处理.md)**
- **[ISSUE规范](#ISSUE规范.md)**
- **[提出PR](#提出PR.md)**



<h2 id="代码风格.md">代码风格</h2>

请遵循以下编码风格，以使得Triton Ascend易于开发、维护和审查。

- 编码指南

  请使用Triton Ascend社区统一的编码风格，python建议的编码风格是[PEP 8编码样式](https://pep8.org/)，C++编码所建议的风格是  [LLVM 编码规范](https://llvm.org/docs/CodingStandards.html))  。可以使用[clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)，[CppLint](https://github.com/cpplint/cpplint)，[CppCheck](http://cppcheck.sourceforge.net/)，[CMakeLint](https://github.com/cmake-lint/cmake-lint)，[CodeSpell](https://github.com/codespell-project/codespell)，[ShellCheck](https://github.com/koalaman/shellcheck)和[pylint](https://pylint.org/)检查代码的格式，建议在您的IDE中安装这些插件。

- 单元测试指南

  请使用Triton Ascend社区统一的单元测试风格，python建议的单元测试风格是[pytest](http://www.pytest.org/en/latest/)，C++建议的单元测试风格是[Googletest Primer](#https://github.com/google/googletest/blob/master/docs/primer.md)。测试用例的设计意图应该通过它的注释名称来反映。测试用例的设计请参考[gather测试用例](https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/pytest_ut/test_gather.py)，[layer_norm测试用例](https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/tutorials/03-layer-norm.py)

- 重构指南

  我们鼓励开发人员对我们的代码进行重构来消除【代码坏味道】。重构的代码也应该遵循编码风格和测试风格的要求。当您收到警告时，您需要重构要合并的代码。



<h2 id="Fork-Pull开发模式.md">Fork-Pull开发模式</h2>

1、Fork Triton Ascend项目

在您向Triton Ascend项目提交自己的代码之前，请确保已经将Triton Ascend项目Fork到您自己的存储库。后续您将在自己Fork的项目上进行开发，并通过Pull Request的方式合并到Triton Ascend项目。这意味着Triton Ascend存储库和您自己的存储库之间存在并行开发，因此请注意保持存储库之间的一致性。

2、克隆远程仓库

使用git克隆您fork的Triton Ascend项目&添加上游仓库upstream：

```shell
git clone https://gitcode.com/{your_forked_repo}/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
git remote add upstream https://gitcode.com/Ascend/triton-ascend.git
```

3、本地环境开发代码

在开发您的代码之前，您需要根据[Triton Ascend安装指南](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/installation_guide.md)搭建开发环境。

为避免多个分支间的不一致问题，请创建新的本地开发分支进行新特性的开发：

```shell
git checkout -b {new_branch_name} origin/master
git fetch upstream #Fetch the latest code from the upstream repository
git rebase upstream/master #Rebase onto the latest upstream
```

以master分支为例，Triton Ascend可能会根据需要创建版本分支或下游开发分支。当您创建完分支&同步上游master分支更新后，就可以开始开发您的代码了。

4、代码更改自测

完成代码更改后，请检查您的更改是否可以通过测试：

在本地代码分支的ascend/examples/pytest_ut路径下为您开发的代码编写测试用例代码，并在本地环境中验证您的测试脚本，确保您的更改可以通过测试。

5、代码推送到远程仓库

代码更新&测试完成后，推送您的commit到您的远程仓库。

```shell
git add .
git status #Check the updated files
git commit -m "Your commit title"
git commit -s --amend #Add the concrete description of your commit
git push origin {your_new_branch_name}
```

6、向Triton Ascend主仓创建拉取请求

代码推送至您的远程仓库后，您需要在您的新分支和Triton Ascend master分支之间新建Pull Request。完成新建合并请求后，“Jenkins CI“将自动设置为您构建流水线测试。您的Pull Request请尽快合并到上游master分支，以降低合并风险。

提交PR后流水线执行命令流程

- 如果PR的标签显示ascend-cla/no，签署cla后评论/check-cla检查cla签署状态，cla签署成功后获得标签 ascend-cla/yes。

  ```shell
  /check-cla 
  ```

- 评论/compile启动流水线测试，如果未通过测试，根据提示修改后再次评论/compile触发流水线测试，通过后获得标签 ci-pipeline-passed。

  ```shell
  /compile
  ```

- 如果SC-FAIL，检查修改后可评论compile#openlibing手动触发检查，检查通过后获得标签 SC-SUCC。

  ```tex
  compile#openlibing
  ```

- 流水线pass之后（收到ci-pipeline-passed、ascend-cla/yes、SC-SUCC标签），根据提示@committers进行代码review，以便快速合入。



<h2 id="代码门禁异常处理.md">代码门禁异常处理</h2>

代码门禁异常主要包含以下几种情况，请根据相关提示信息解决门禁异常问题。

- 编译失败

  请根据提示信息，检查编译失败的原因，解决后重新编译即可 。

- 静态检查失败

  请根据提示信息，查找出代码中的异常信息并解决。

- CI流水线未通过

  请根据提示信息，查找出CI流水线未通过的测试用例并检查原因，解决后重新运行CI流水线。
  
  

<h2 id="ISSUE规范.md">ISSUE规范</h2>

为项目做贡献的一个好的方法是在遇到问题时发送详细报告。我们总是非常感谢写得详细、彻底的错误报告，并会因此非常感谢您！

在报告问题时，请参考以下格式：

- 您环境里使用的软件版本（Triton Ascend、python、os等）？
- 这是一个错误报告还是功能请求？
- 您报告的是什么样的问题，添加对应的标签以便在问题仪表盘上突出显示？
- 发生了什么？
- 您预计会发生什么？
- 如何重现它？（尽可能精确）

不同类别的ISSUE填写模板请参考[ISSUE填写规范](https://gitcode.com/Ascend/triton-ascend/issues/create/choose)

问题咨询：

- 如果您发现一个未解决的问题，而这个问题正是您要解决的，请对该问题发表评论，告诉其他人您将负责这个问题。
- 如果问题已经打开一段时间，请您在解决该问题前进行预检查。
- 如果您解决了自己报告的问题，在关闭该问题前还需要让其他人知道。



<h2 id="提出PR.md">提出PR</h2>

- 在[GitCode](https://gitcode.com/Ascend/triton-ascend)上提出您的想法作为问题。
- 如果要开发的新功能需要大量设计细节，您还应提交设计方案。
- 在问题讨论和设计方案审查达成共识后，再进行fork开发并提交PR。
- 在从Approver那里收到2+LGTM（Looks Good To Me）前不允许任何PR 。请注意审批人不允许在自己的PR上添加LGTM。
- 在PR被充分讨论后，将根据讨论结果对PR进行合并、拒绝或放弃。
- PR样例：[PR样例](https://gitcode.com/Ascend/triton-ascend/pull/936)

### 注意事项：

-   应避免任何不相关的更改。
-   确保您的提交历史是简洁有序的。
-   创建PR前请rebase上游仓库最新代码。
-   对于错误修复 PR，请确保链接所有相关Issue 和 PR。
