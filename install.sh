#!/bin/sh
# sference CLI installer (macOS / Linux)
# Usage: curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh | sh
#
# Optional environment:
#   SFERENCE_CLI_VERSION=0.0.7   pin PyPI version (default: latest)
#   SFERENCE_NO_UV_BOOTSTRAP=1   fail instead of installing uv when missing
set -eu

PACKAGE="sference-cli"
BINARY="sference"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
DEFAULT_BIN_DIR="${HOME}/.local/bin"

if [ -t 1 ]; then
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
	printf "${BOLD}%s${RESET}\n" "$1"
}

success() {
	printf "${GREEN}%s${RESET}\n" "$1"
}

warn() {
	printf "${YELLOW}%s${RESET}\n" "$1"
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

bootstrap_uv() {
	if command -v uv >/dev/null 2>&1; then
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
	ensure_path
	command -v uv >/dev/null 2>&1 || error "uv installation failed"
}

install_with_uv() {
	VERSION_SPEC=""
	if [ -n "${SFERENCE_CLI_VERSION:-}" ]; then
		VERSION_SPEC="==${SFERENCE_CLI_VERSION}"
	fi
	info "Installing ${PACKAGE}${VERSION_SPEC} with uv..."
	uv tool install "${PACKAGE}${VERSION_SPEC}" --force
}

install_with_pipx() {
	if ! command -v pipx >/dev/null 2>&1; then
		return 1
	fi
	VERSION_SPEC=""
	if [ -n "${SFERENCE_CLI_VERSION:-}" ]; then
		VERSION_SPEC="==${SFERENCE_CLI_VERSION}"
	fi
	info "Installing ${PACKAGE}${VERSION_SPEC} with pipx..."
	pipx install "${PACKAGE}${VERSION_SPEC}" --force
}

python310_or_newer() {
	PY="$1"
	"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

install_with_pip() {
	PY=""
	for candidate in python3 python; do
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
	info "Installing ${PACKAGE}${VERSION_SPEC} with ${PY} -m pip --user..."
	"$PY" -m pip install --user "${PACKAGE}${VERSION_SPEC}"
}

warn_path() {
	case ":${PATH}:" in
	*":${DEFAULT_BIN_DIR}:"*) return 0 ;;
	esac
	echo ""
	warn "Add ${DEFAULT_BIN_DIR} to your PATH:"
	echo "  export PATH=\"${DEFAULT_BIN_DIR}:\$PATH\""
	echo ""
	echo "Add that line to ~/.bashrc, ~/.zshrc, or your shell profile, then open a new shell."
}

verify_install() {
	ensure_path
	if ! command -v "$BINARY" >/dev/null 2>&1; then
		warn_path
		error "${BINARY} was installed but is not on PATH."
	fi
	INSTALLED_VERSION="$("$BINARY" --version 2>/dev/null || true)"
	if [ -n "$INSTALLED_VERSION" ]; then
		success "Installed ${INSTALLED_VERSION}"
	else
		success "Installed ${BINARY} to $(command -v "$BINARY")"
	fi
}

main() {
	if [ -t 1 ]; then
		echo ""
		echo "  ╔══════════════════════════════════════╗"
		echo "  ║           sference CLI               ║"
		echo "  ╚══════════════════════════════════════╝"
		echo ""
	else
		info "sference CLI installer"
		echo ""
	fi

	OS="$(uname -s 2>/dev/null || true)"
	case "$OS" in
	Darwin | Linux) ;;
	*) error "Unsupported operating system: ${OS:-unknown}. Use install.ps1 on Windows." ;;
	esac

	ensure_path

	if ! command -v uv >/dev/null 2>&1; then
		if [ "${SFERENCE_NO_UV_BOOTSTRAP:-}" = "1" ]; then
			warn "uv not found; trying pipx or pip..."
		else
			bootstrap_uv
		fi
	fi

	if command -v uv >/dev/null 2>&1; then
		install_with_uv
		echo ""
		verify_install
		print_get_started
		return 0
	fi

	if install_with_pipx; then
		echo ""
		verify_install
		print_get_started
		return 0
	fi

	if install_with_pip; then
		echo ""
		verify_install
		print_get_started
		return 0
	fi

	error "Could not install ${PACKAGE}. Install uv (https://docs.astral.sh/uv/) or Python 3.10+, then retry."
}

print_get_started() {
	echo ""
	info "Get started:"
	echo "  sference auth login              # Authenticate via browser"
	echo "  sference auth login --api-key    # Authenticate with an API key"
	echo "  sference batch submit --help     # Submit a batch job"
	echo "  sference --help                  # See all commands"
	echo ""
}

main
