# ACROSS Tools Contributing Guide

So you'd like to contribute to the ACROSS Tools? Below are some guidelines for contributors to follow. Contributions come in all shapes and sizes. We appreciate your help with documentation, unit tests, framework code, continuous-integration, or simply reporting bugs and improvement ideas. We can't promise that we'll accept every suggestion or fix every bug in a timely manner but we'll respond to you as quickly as possible.

<!-- vscode-markdown-toc -->
- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
  - [Bug Reports](#bug-reports)
    - [Before Reporting a Bug](#before-reporting-a-bug)
    - [Reporting a Bug](#reporting-a-bug)
    - [What Happens to my Bug Report?](#what-happens-to-my-bug-report)
  - [New Feature Requests](#new-feature-requests)
    - [Before Requesting a New Feature](#before-requesting-a-new-feature)
    - [Requesting a New Feature](#requesting-a-new-feature)
    - [What Happens to my Feature Request?](#what-happens-to-my-feature-request)
  - [Pull Requests](#pull-requests)
    - [Before starting your Pull Request](#before-starting-your-pull-request)
    - [Creating a Pull Request](#creating-a-pull-request)
    - [What Happens to My Pull Request?](#what-happens-to-my-pull-request)

## <a name='CodeofConduct'></a>Code of Conduct

This project and everyone participating in it is governed by the [ACROSS Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the product team.

## <a name='WaystoContribute'></a>Ways to Contribute

### <a name='BugReports'></a>Bug Reports

#### <a name='BeforeReportingaBug'></a>Before Reporting a Bug
1. Perform a cursory search to see if the bug has already been reported. If a bug has been reported and the issue is still open, add a comment to the existing issue instead of opening a new one.

#### <a name='ReportingaBug'></a>Reporting a Bug

If you find a bug in our code don't hesitate to report it:

1. Open an issue using the [bug report template](https://github.com/NASA-ACROSS/across-tools/issues/new?template=bug.yaml).
2. Describe the issue.
3. Describe the expected behavior if the bug did not occur.
4. Provide the reproduction steps that someone else can follow to recreate the bug or error on their own.
5. If applicable, add code snippets or references to the software.
6. Provide the system the bug was observed on including the hardware, operating system, and versions.
7. Provide any additional context if applicable.
8. Provide your full name or GitHub username and your company organization if applicable.

#### <a name='WhatHappenstomyBugReport'></a>What Happens to my Bug Report?

1. The ACROSS team will label the issue.
2. A team member will try to reproduce the issue with your provided steps. If the team is able to reproduce the issue, the issue will be left to be implemented by someone.

### <a name='NewFeatureRequests'></a>New Feature Requests

ACROSS has a multitude of users from different fields and backgrounds. We appreciate your ideas for enhancements! 

#### <a name='RequestingaNewFeature'></a>Requesting a New Feature

1. Open an issue using the feature request template.
2. Describe the feature.
3. Describe the solution you would like.
4. Describe alternatives you've considered.
5. Provide any additional context if applicable.
6. Provide your full name or GitHub username and your company organization if applicable.

#### <a name='WhatHappenstomyFeatureRequest'></a>What Happens to my Feature Request?

1. The project team will label the issue.
2. The project team will evaluate the feature request, possibly asking you more questions to understand its purpose and any relevant requirements. If the issue is closed, the team will convey their reasoning and suggest an alternative path forward.
3. If the feature request is accepted, it will be marked for implementation.

### <a name='PullRequests'></a>Pull Requests

#### <a name='BeforestartingyourPullRequest'></a>Before starting your Pull Request

Ready to Add Your Code? Follow GitHub's fork-branch-pull request pattern.

1. Fork the across-tools repository.

2. Find the related issue number or create an associated issue that explains the intent of your new code. 

3. Create a new branch in your fork to work on your fix. We recommend naming your branch `[fix/feat]:ISSUE_NUMBER-<FIX_SUMMARY>`.

3. Add commits to your branch. For information on commit messages, review [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary).

#### <a name='CreatingaPullRequest'></a>Creating a Pull Request

We recommend creating your pull-request as a "draft" and to commit early and often so the community can give you feedback at the beginning of the process as opposed to asking you to change hours of hard work at the end.

1. For the title, use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary).
2. Describe the contribution. First document which issue number was fixed using the template "Fix #XYZ". Then describe the contribution.
3. Provide what testing was used to confirm the pull request resolves the link issue. If writing new code, also provide the associated coverage unit tests. 
4. Provide the expected behavior changes of the pull request.
6. Provide any additional context if applicable.
7. Verify that the PR passes all workflow checks. If you expect some of these checks to fail. Please note it in the Pull Request text or comments. 

#### <a name='WhatHappenstoMyPullRequest'></a>What Happens to My Pull Request?

1. The ACROSS team will label and evaluate the pull request in their weekly stakeholder meetings
2. If the pull request is accepted, it will be merged into across-tools.
