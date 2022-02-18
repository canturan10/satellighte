#!/bin/bash

set -x
set -e

release_type=${1}
repository_url=$(git config --get remote.origin.url)
old_version=$(python scripts/release.py get_version)

commit=${CI_COMMIT_SHA:-$(git rev-parse HEAD)}
branch=${ALLOWED_RELEASE_BRANCH:-master}

if ! git branch -a --contains "${commit}" | grep -e "^[* ]*remotes/origin/${branch}\$"
then
  echo -e "###\n### Not on ${branch}. Only ${branch} commits can be released.\n###"
  exit 1
else
  echo -e "###\n### Releasing of ${commit} on ${branch}\n###"
fi

git checkout ${branch}

python scripts/release.py inc-${release_type}
new_version=$(python scripts/release.py get_version)

git add .
git commit -m "Incrementing working version to ${new_version} after ${old_version} release."

git tag -a v${new_version} -m "Releasing version v${new_version}"

echo ${new_version}
echo HEAD:${branch}

git push ${repository_url} v${new_version}

git push ${repository_url} HEAD:${branch}