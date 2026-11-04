# Publishing this repository to GitHub

Steps for the first upload of py3wafo, and for later releases.

## 1. Repository path

The documentation points to `wengangmao/py3wafo`. If you publish under another
account or name, for example in a fork, replace that path in:

- `README.md` (badges and install commands)
- `pyproject.toml` (`[project.urls]`)
- `docs/INSTALLATION.md`
- `CONTRIBUTING.md`
- this file

To find every occurrence, run `git grep -n "wengangmao/py3wafo"`.

## 2. Create the local Git repository

GitHub rejects pushes whose commits carry an email address that your account
keeps private (error `GH007`). Commit with your GitHub no-reply address, which
is shown at <https://github.com/settings/emails> and has the form
`ID+username@users.noreply.github.com`. From the `py3wafo` folder:

```bash
git init -b main
git config user.name "Your Name"
git config user.email "ID+username@users.noreply.github.com"
git add .
git status                               # check that no build output or .venv is listed
git commit -m "py3wafo 0.4.0: Python 3 port of WAFO"
```

If a push was already rejected with `GH007`, set the address as above, then
correct the unpushed commits. For a single commit:
`git commit --amend --reset-author --no-edit`.

## 3. Create the GitHub repository and push

On GitHub, create a new **empty** repository named `py3wafo`, without a
README, licence or .gitignore, since this folder already has them. Then:

```bash
git remote add origin https://github.com/wengangmao/py3wafo.git
git push -u origin main
```

With the GitHub CLI, the same can be done in one step:

```bash
gh repo create py3wafo --public --source . --push
```

The repository is about 17 MB; the largest file, `src/wafo/data/yura87.dat`,
is 5.6 MB, well within GitHub's 100 MB per-file limit, so Git LFS is not
needed.

## 4. Check continuous integration

The push starts the workflow in `.github/workflows/ci.yml`. It runs the test
suite on Linux, Windows and macOS for Python 3.10, 3.12 and 3.13, checks the
oldest supported NumPy and SciPy, and builds the distributions. Results are on
the repository's **Actions** tab, and the README badge turns green after the
first successful run.

## 5. Make a release

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/wafo/__init__.py`, and add an entry to `CHANGELOG.md`.
2. Commit, tag and push:

   ```bash
   git commit -am "Release 0.4.0"
   git tag -a v0.4.0 -m "py3wafo 0.4.0"
   git push origin main --tags
   ```

3. Build the distributions and attach them to a GitHub release:

   ```bash
   python -m pip install build
   python -m build
   gh release create v0.4.0 dist/* --title "py3wafo 0.4.0" --notes-file CHANGELOG.md
   ```

Users can then install a fixed version with
`pip install "py3wafo @ git+https://github.com/wengangmao/py3wafo.git@v0.4.0"`.

## 6. Optional: publish on PyPI

Publishing on PyPI lets users run `pip install py3wafo`. Check first that the
name is free at <https://pypi.org/project/py3wafo/>. Then:

```bash
python -m pip install twine
python -m twine upload --repository testpypi dist/*   # trial run on TestPyPI
python -m twine upload dist/*
```

You will need a PyPI account and an API token.
