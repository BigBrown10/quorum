# Release Checklist

Use this checklist before cutting a private release or pushing to the main branch.

## Python Core

- [ ] Run Ruff on `src` and `tests`
- [ ] Run MyPy on `src` and `tests`
- [ ] Run the Python test suite
- [ ] Build the Python package with `python -m build`

## TypeScript Client

- [ ] Run `npm install` in `clients/ts`
- [ ] Run `npm run build` in `clients/ts`

## Repository Hygiene

- [ ] Confirm there are no placeholder or demo comments left in code or docs
- [ ] Confirm `IMPLEMENTATION_PLAN.md` reflects the current status
- [ ] Confirm the CI workflow passes locally or in GitHub Actions
- [ ] Confirm private release notes are ready if needed

## Push Readiness

- [ ] Tag or branch the repository for the first private push
- [ ] Review the final diff for accidental leaks or noisy debug output
