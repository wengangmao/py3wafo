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

## 5. One-time setup: let GitHub publish to PyPI

Releases are uploaded to PyPI by `.github/workflows/publish.yml` using
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts
this repository's workflow directly, so no password or API token is stored in
GitHub. Set it up once:

1. **PyPI.** Sign in at <https://pypi.org>, open
   **Your projects → py3wafo → Manage → Publishing**, and add a GitHub
   publisher with:

   | Field | Value |
   |---|---|
   | Owner | `wengangmao` |
   | Repository name | `py3wafo` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   For a project that does not exist on PyPI yet, use
   <https://pypi.org/manage/account/publishing/> to add a "pending" publisher
   under the new project name instead.

2. **TestPyPI (optional, for trial uploads).** Repeat at
   <https://test.pypi.org/manage/account/publishing/> as a pending publisher
   for project `py3wafo`, with environment name `testpypi`. TestPyPI uses a
   separate account from PyPI.

3. **GitHub environments.** In the repository, open
   **Settings → Environments** and create environments named `pypi` and
   `testpypi`. Under `pypi`, optionally add yourself as a required reviewer,
   so every upload waits for your approval.

## 6. Make a release

1. Set the new version in `src/wafo/__init__.py` (`__version__`); the build
   reads it from there. Add a section for it to `CHANGELOG.md`. PyPI never
   accepts the same version twice, even after a file is deleted.
2. Commit and push, and wait for CI to pass:

   ```bash
   git commit -am "Release 0.5.0"
   git push
   ```

3. Optional trial run: on the **Actions** tab, choose **Publish to PyPI →
   Run workflow**, target `testpypi`, then check
   <https://test.pypi.org/project/py3wafo/>. TestPyPI also rejects repeated
   versions, so a trial uses up that version number there, but not on PyPI.

4. Create the GitHub release. The tag must be `v` followed by the version:

   ```bash
   gh release create v0.5.0 --title "py3wafo 0.5.0" --generate-notes
   ```

   You can do the same on GitHub under **Releases → Draft a new release**.

5. Publishing the release starts the workflow. It builds the sdist and
   wheel, checks that the tag matches the version, runs the full test suite
   on the built wheel, and uploads to PyPI. Follow it on the **Actions** tab;
   if you added a reviewer, approve the `pypi` deployment there.

6. Check the result:

   ```bash
   python -m pip install --upgrade "py3wafo[all]"
   python -c "import wafo; print(wafo.__version__)"
   ```

If a step fails before the upload, nothing reaches PyPI: fix the problem,
delete the release and its tag on GitHub, and release again under the same
version.

## 7. Manual upload (fallback)

Without the workflow, upload from your own computer with an API token from
<https://pypi.org/manage/account/token/>:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*                 # user name: __token__, password: the token
```

Add `--repository testpypi` to the upload command for a trial on TestPyPI.
