# If you have `direnv` loaded in your shell, and allow it in the repository,
# the `make` command will point at the `scripts/make` shell script.
# This Makefile is just here to allow auto-completion in the terminal.

actions = \
	clean \
	help \
	setup

.PHONY: $(actions)
$(actions):
	@python scripts/make "$@"
