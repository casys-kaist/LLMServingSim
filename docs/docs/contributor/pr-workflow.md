---
sidebar_position: 6
title: PR workflow
---

# PR workflow

How a contribution gets from your fork into `main`. Read this once;
afterward each PR should take you about ten minutes of process
overhead on top of the actual work.

## Branch model

- **`main`**: the active development branch. All PRs land here.
- **Artifact branches**: per-paper reproducibility branches, named
  after the venue (e.g., `ispass26-artifact`). **Don't open PRs
  against these.** They are frozen at the artifact submission state.
- **Your work**: a feature branch off `main`, named descriptively
  (`add-deepseek-v3`, `fix-evict-accumulation`, `docs-cluster-config`).
  Don't push to `main` directly even if you have permissions.

```bash
git checkout main
git pull
git checkout -b add-deepseek-v3
```

## Commit hygiene

- **Short imperative one-liner.** Same style as the existing log:
  `Fix incorrect evict_size accumulation`, `Add Qwen3 model
  support`, `Document MoE expert routing`.
- **One logical change per commit.** A refactor and a feature in
  the same commit is a reviewer's nightmare.
- **Don't amend published commits.** If you pushed it, follow up
  with a new commit. Force-pushing your branch is fine *before*
  review starts, generally not after.
- **No `--no-verify`** to bypass pre-commit hooks. Fix what failed.
- **No `Co-authored-by`** unless someone really did pair-program
  with you on this commit.

A good commit message:

```
Fix evict_size accumulation when prefix cache spills to CPU

Spilling counted the block twice: once in the NPU eviction and
again when the second-tier pool inserted it. Drop the second
increment; the test in single_node_memory_instance.json now
matches the bench reference.
```

A bad one:

```
fixes
```

## Before you push

Run through the checklist:

1. **Smoke run passes.** See **[Validating your
   changes](./validating-changes)**, step 1.
2. **Targeted scenarios pass** for whatever you touched. Step 2.
3. **Bench validation hasn't regressed** if your change affects
   end-to-end accuracy. Step 3.
4. **Conventions checklist**: `getattr` fallbacks, `head_dim`
   handling, English-only, layer names, no `astra-sim/inputs/`
   edits. See **[Coding conventions](./conventions)**.
5. **Docs updated** if behavior changed. The relevant page under
   `docs/`, plus the module's `README.md` if applicable.
6. **No machine-specific paths or generated files** in the diff.
   Sanity-check with `git diff --stat` and
   `git diff --check`.

## Opening the PR

Push to your fork (or branch if you have direct access):

```bash
git push -u origin add-deepseek-v3
```

Then open the PR against `casys-kaist/LLMServingSim:main`. The
description should include:

```
## What this changes

A 1-3 sentence summary of the user-visible change.

## Why

The motivation: the bug it fixes, the feature it enables, the
research question it lets you ask.

## Validation

The exact command(s) you ran and the key result. For example:

  ./bench/examples/validate.sh Llama-3.1-8B
  -> TTFT MAPE 2.1% (was 2.3%), TPOT 1.7% (unchanged)

## Notes

Anything subtle: known limitations, related issues, follow-ups
you intentionally did not include.
```

You don't need a heavy template. The validation section is the one
non-negotiable part: it gives the reviewer something concrete to
rerun and gives the git log a record of what was checked.

## What review looks like

- **Initial response**: usually within 2-3 days for the first round.
  Time-zone overlap with KAIST (UTC+9) helps but isn't required.
- **Reviewers**: at least one of the main contributors
  ([@JaehongCho](https://github.com/JaehongCho),
  [@hmchoi](https://github.com/hmchoi)) plus whoever owns the touched
  area. For docs-only PRs, one approval is enough.
- **What gets blocked vs. nit-picked**:
  - **Blockers**: bench regressions beyond ~5%, broken smoke run,
    convention violations from the "never do this" list, missing
    docs for new flags.
  - **Nits**: naming, code style preferences, doc phrasing. The
    reviewer will say "nit:" or use the GitHub label. Address them
    if you agree; defer with a sentence if you don't.
- **Conversation style**: terse and direct. "This won't work for
  MoE" is not a personal attack; it's faster than the polite
  version. Reply in kind.

## Squash, rebase, or merge?

The project squashes most PRs to a single commit on `main`, with
the PR title becoming the commit message. You don't need to clean
up your branch's intermediate commits beforehand. If your PR is
genuinely best as multiple commits (e.g., a refactor + a feature
that depends on it), say so in the description and a maintainer
will rebase rather than squash.

## Attribution

External contributors get credit in two places:

1. **GitHub commit history**: your authorship is preserved on
   merge.
2. **`CONTRIBUTORS.md`**: the maintainer adds a line with your
   GitHub handle and a link to the PR or issue — under
   "Code" for a merged patch, under "Reports and analysis" for an
   issue that pinned down a real problem. Reports get their own
   section rather than a footnote, and the changelog entry for the
   fix names you too.

You don't need to add yourself to the contributors list in your
PR. The maintainer adds it on merge.

## After merge

- **Pull `main`** before starting the next change. Your local
  branch is no longer authoritative.
- **Delete the merged branch** locally and on the remote
  (GitHub offers a button after merge; `git branch -d add-deepseek-v3`
  locally).
- **Watch CI on `main` for a day or two**. If something broke that
  the PR didn't catch, you're best positioned to fix it quickly.

## When things go wrong

- **My PR sat for a week with no reviews.** Ping the PR with a
  one-liner. Maintainers do miss notifications.
- **A reviewer requested changes I disagree with.** Explain your
  reasoning in a comment. If you still disagree after the
  reviewer's reply, escalate by tagging the other main contributor
  for a tiebreaker. We'd rather have the discussion than land the
  wrong design.
- **My change regressed bench beyond what I expected.** Don't merge
  it. Open the PR as a draft and tag the regression in the
  description; we'll figure out together whether it's a bug in your
  change, in the existing baseline, or in the validation
  methodology.
- **I broke something on `main`.** It happens. Open a follow-up PR
  with a `Fix ...` commit; don't `git push --force` to `main`.

## What's next

You've got the full picture now. Go pick a starter issue or open
a new one with `[contributor]` in the title to discuss what you'd
like to work on.

Welcome aboard.
