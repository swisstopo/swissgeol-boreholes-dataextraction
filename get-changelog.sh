#!/bin/bash
# This script gets all changelog entries from CHANGELOG.md since last release.

set -e

tempDir="$(mktemp -d)"
tempFile=$tempDir/gh_release_notes.md

# Get changelog entries since last release
cat CHANGELOG.md | \
  grep -Pazo '(?s)(?<=\#{2} \[Unreleased\]\n{2}).*?(?=\n\#{2} v|$)' \
  > $tempFile

# Improve readability and add some icons
sed -i -E 's/(###) (Added)/\1 🚀 \2/' $tempFile
sed -i -E 's/(###) (Changed)/\1 🔨 \2/' $tempFile
sed -i -E 's/(###) (Fixed)/\1 🐛 \2/' $tempFile
sed -i 's/\x0//g' $tempFile

cat $tempFile

# Cleanup temporary files
trap 'rm -rf -- "$tempDir"' EXIT