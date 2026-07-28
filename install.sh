#!/bin/sh
# sference CLI installer (macOS / Linux)
#
# Recommended (updates PATH in the current Terminal window):
#   eval "$(curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh)"
#
# Legacy (installs for new shells only; same window needs eval or a new tab):
#   curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh | sh
#
# Optional environment:
#   SFERENCE_CLI_VERSION=0.0.7   pin PyPI version (default: latest)
#   SFERENCE_NO_UV_BOOTSTRAP=1   fail instead of installing uv when missing
set -eu

PACKAGE="sference-cli"
BINARY="sference"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
DEFAULT_BIN_DIR="${HOME}/.local/bin"
PATH_MARKER="# sference-install: add ~/.local/bin to PATH"
PATH_UPDATED=0
UV_BIN=""

if [ -t 2 ]; then
	BOLD='\033[1m'
	GREEN='\033[0;32m'
	RED='\033[0;31m'
	YELLOW='\033[0;33m'
	RESET='\033[0m'
else
	BOLD=''
	GREEN=''
	RED=''
	YELLOW=''
	RESET=''
fi

info() {
	printf "${BOLD}%s${RESET}\n" "$1" >&2
}

success() {
	printf "${GREEN}%s${RESET}\n" "$1" >&2
}

warn() {
	printf "${YELLOW}%s${RESET}\n" "$1" >&2
}

error() {
	printf "${RED}Error: %s${RESET}\n" "$1" >&2
	exit 1
}

ensure_path() {
	case ":${PATH}:" in
	*":${DEFAULT_BIN_DIR}:"*) ;;
	*) PATH="${DEFAULT_BIN_DIR}:${PATH}" ;;
	esac
	export PATH
}

resolve_uv_bin() {
	if [ -n "${UV_BIN}" ] && [ -x "${UV_BIN}" ]; then
		return 0
	fi
	if command -v uv >/dev/null 2>&1; then
		UV_BIN=$(command -v uv)
		return 0
	fi
	if [ -x "${DEFAULT_BIN_DIR}/uv" ]; then
		UV_BIN="${DEFAULT_BIN_DIR}/uv"
		return 0
	fi
	UV_BIN=""
	return 1
}

profile_contains_local_bin() {
	profile="$1"
	[ -f "$profile" ] || return 1
	if grep -qF "$PATH_MARKER" "$profile" 2>/dev/null; then
		return 0
	fi
	if grep -qF '.local/bin' "$profile" 2>/dev/null; then
		return 0
	fi
	return 1
}

append_path_block() {
	profile="$1"
	if profile_contains_local_bin "$profile"; then
		return 0
	fi
	{
		echo ""
		echo "$PATH_MARKER"
		printf '%s\n' "export PATH=\"${DEFAULT_BIN_DIR}:\$PATH\""
	} >> "$profile"
	PATH_UPDATED=1
}

persist_shell_path() {
	mkdir -p "${DEFAULT_BIN_DIR}"
	OS="$(uname -s 2>/dev/null || true)"
	if [ "$OS" = "Darwin" ]; then
		for profile in "${HOME}/.zprofile" "${HOME}/.zshrc"; do
			append_path_block "$profile"
		done
		return 0
	fi
	for profile in "${HOME}/.profile" "${HOME}/.bash_profile" "${HOME}/.bashrc"; do
		append_path_block "$profile"
	done
}

bootstrap_uv() {
	if resolve_uv_bin; then
		return 0
	fi
	if [ "${SFERENCE_NO_UV_BOOTSTRAP:-}" = "1" ]; then
		error "uv is not installed. Install uv (https://docs.astral.sh/uv/) or unset SFERENCE_NO_UV_BOOTSTRAP."
	fi
	info "Installing uv (Python toolchain manager)..."
	if command -v curl >/dev/null 2>&1; then
		curl -LsSf "$UV_INSTALL_URL" | sh
	elif command -v wget >/dev/null 2>&1; then
		wget -qO- "$UV_INSTALL_URL" | sh
	else
		error "Neither curl nor wget found. Install one of them and retry."
	fi
	mkdir -p "${DEFAULT_BIN_DIR}"
	persist_shell_path
	ensure_path
	resolve_uv_bin || error "uv installation failed"
}

install_with_uv() {
	VERSION_SPEC=""
	if [ -n "${SFERENCE_CLI_VERSION:-}" ]; then
		VERSION_SPEC="==${SFERENCE_CLI_VERSION}"
	fi
	resolve_uv_bin || error "uv is not available"
	mkdir -p "${DEFAULT_BIN_DIR}"
	export UV_TOOL_BIN_DIR="${DEFAULT_BIN_DIR}"
	info "Installing ${PACKAGE}${VERSION_SPEC} with uv..."
	"$UV_BIN" tool install "${PACKAGE}${VERSION_SPEC}" --force
	info "Installing mitmproxy (for 'sference launch claude' proxy mode) with uv..."
	"$UV_BIN" tool install mitmproxy --force || warn "mitmproxy install failed; 'sference launch claude' will need --no-anthropic"
}

install_with_pipx() {
	if ! command -v pipx >/dev/null 2>&1; then
		return 1
	fi
	VERSION_SPEC=""
	if [ -n "${SFERENCE_CLI_VERSION:-}" ]; then
		VERSION_SPEC="==${SFERENCE_CLI_VERSION}"
	fi
	mkdir -p "${DEFAULT_BIN_DIR}"
	info "Installing ${PACKAGE}${VERSION_SPEC} with pipx..."
	pipx install "${PACKAGE}${VERSION_SPEC}" --force
	info "Installing mitmproxy (for 'sference launch claude' proxy mode) with pipx..."
	pipx install mitmproxy --force || warn "mitmproxy install failed; 'sference launch claude' will need --no-anthropic"
}

python310_or_newer() {
	PY="$1"
	"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

install_with_pip() {
	PY=""
	for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
		if command -v "$candidate" >/dev/null 2>&1 && python310_or_newer "$candidate"; then
			PY="$candidate"
			break
		fi
	done
	if [ -z "$PY" ]; then
		return 1
	fi
	VERSION_SPEC=""
	if [ -n "${SFERENCE_CLI_VERSION:-}" ]; then
		VERSION_SPEC="==${SFERENCE_CLI_VERSION}"
	fi
	mkdir -p "${DEFAULT_BIN_DIR}"
	info "Installing ${PACKAGE}${VERSION_SPEC} with ${PY} -m pip --user..."
	"$PY" -m pip install --user "${PACKAGE}${VERSION_SPEC}"
	info "Installing mitmproxy (for 'sference launch claude' proxy mode) with ${PY} -m pip --user..."
	"$PY" -m pip install --user mitmproxy || warn "mitmproxy install failed (needs Python >=3.12); 'sference launch claude' will need --no-anthropic"
}

resolve_binary_path() {
	if [ -x "${DEFAULT_BIN_DIR}/${BINARY}" ]; then
		printf '%s\n' "${DEFAULT_BIN_DIR}/${BINARY}"
		return 0
	fi
	ensure_path
	if command -v "$BINARY" >/dev/null 2>&1; then
		command -v "$BINARY"
		return 0
	fi
	return 1
}

# stdout only — picked up by: eval "$(curl -fsSL .../install.sh)"
emit_path_export() {
	printf 'export PATH="%s:$PATH"\n' "$DEFAULT_BIN_DIR"
}

verify_install() {
	persist_shell_path
	ensure_path
	BIN_PATH=""
	if ! BIN_PATH=$(resolve_binary_path); then
		error "${BINARY} was not installed to ${DEFAULT_BIN_DIR}/${BINARY}. Check errors above and retry."
	fi
	INSTALLED_VERSION="$("$BIN_PATH" --version 2>/dev/null || true)"
	if [ -n "$INSTALLED_VERSION" ]; then
		success "Installed ${INSTALLED_VERSION}"
	else
		success "Installed ${BINARY} at ${BIN_PATH}"
	fi
	if [ "$PATH_UPDATED" -eq 1 ]; then
		warn "Added ${DEFAULT_BIN_DIR} to your shell startup files for future Terminal sessions."
	fi
}

print_get_started() {
	echo "" >&2
	info "Get started:"
	echo "  ${BINARY} auth login              # Authenticate via browser" >&2
	echo "  ${BINARY} auth login --api-key    # Authenticate with an API key" >&2
	echo "  ${BINARY} batch submit --help     # Submit a batch job" >&2
	echo "  ${BINARY} --help                  # See all commands" >&2
	echo "" >&2
}

finish_success() {
	verify_install
	print_get_started
	# Must be last: eval "$(curl ...)" runs this export in the caller's shell.
	emit_path_export
}

main() {
	if [ -t 2 ]; then
		echo "" >&2
		echo "  ╔══════════════════════════════════════╗" >&2
		echo "  ║           sference CLI               ║" >&2
		echo "  ╚══════════════════════════════════════╝" >&2
		echo "" >&2
	else
		info "sference CLI installer"
		echo "" >&2
	fi

	OS="$(uname -s 2>/dev/null || true)"
	case "$OS" in
	Darwin | Linux) ;;
	*) error "Unsupported operating system: ${OS:-unknown}. Use install.ps1 on Windows." ;;
	esac

	mkdir -p "${DEFAULT_BIN_DIR}"
	ensure_path

	if ! resolve_uv_bin; then
		if [ "${SFERENCE_NO_UV_BOOTSTRAP:-}" = "1" ]; then
			warn "uv not found; trying pipx or pip..."
		else
			bootstrap_uv
		fi
	fi

	if resolve_uv_bin; then
		install_with_uv
		finish_success
		return 0
	fi

	if install_with_pipx; then
		finish_success
		return 0
	fi

	if install_with_pip; then
		finish_success
		return 0
	fi

	error "Could not install ${PACKAGE}. Install uv (https://docs.astral.sh/uv/) or Python 3.10+, then retry."
}

main
