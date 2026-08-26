# Agent Guidance

## Change Scope

- Prefer the smallest change that completely solves the requested problem.
- Keep each change focused on one behavioral change or one clearly bounded refactor.
- Before introducing a new abstraction, search for an existing implementation that can be extended or generalized.
- Do not refactor, rename, reformat, or clean up adjacent code unless the requested change requires it.
- Do not broaden a change to fix related issues unless they are necessary for correctness or explicitly requested.
- When prerequisite refactoring is substantial, separate it from the behavioral change into its own commit or PR when practical.

## Reviewability

- Optimize for a diff a reviewer can reason about end to end, not for an arbitrary line-count target.
- Add or update focused tests that demonstrate the behavior being changed; avoid unrelated test churn.
- If a broad diff is unavoidable, explain why the scope cannot be split safely.
- Do not duplicate logic or degrade the design merely to make a diff smaller.
