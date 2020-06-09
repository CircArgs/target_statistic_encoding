#!/usr/bin/env bash


echo "Installing poetry"
pip3 install poetry;
echo "Poetry install"
pipenv install
echo "Running pytest"
poetry run python -m pytest --cov=.
# poetry run python -m pytest --cov=. >> $GITHUB_WORKSPACE/coverage_summary
# echo "Formatting Repo Coverage Badge"
# pipenv run python .github/actions/testing/make_coverage_badge.py
# mkdir badges
# cd badges
# git config --global user.email "$EMAIL"
# git config --global user.name "CircArgs"
# git clone https://github.com/CircArgs/EoPI.git .
# git checkout badges
# cp $GITHUB_WORKSPACE/coverage_badge.svg .
# cp $GITHUB_WORKSPACE/coverage_summary ./pytest_summary.txt
# git config remote.origin.url https://CircArgs:$GITPASS@github.com/CircArgs/EoPI.git
# git add .
# git_hash=$(git rev-parse --short "$GITHUB_SHA")
# git_branch=${GITHUB_REF##*/}
# git commit -m "push from action of $git_branch with commit hash $git_hash"
# git push 

# cat $GITHUB_WORKSPACE/coverage_summary