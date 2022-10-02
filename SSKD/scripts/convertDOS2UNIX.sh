dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
for entry in "$dir"/*
do
  dos2unix "$entry"
done