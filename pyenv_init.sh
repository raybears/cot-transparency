if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
  if command pyenv virtualenv --help 1>/dev/null 2>&1; then
    eval "$(pyenv virtualenv-init -)"
  fi
  SUB='.pyenv/shims'
  if [[ "$PATH" != *"$SUB"* ]]; then
    eval "$(pyenv init --path)"
  fi
fi

