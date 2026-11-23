#!/usr/bin/env bash
# Keep the WSL distro alive while a given process runs.
#
# WSL powers the distro off as soon as the last Windows-side wsl.exe client exits;
# tmux, nohup and systemd linger do not count as clients. This starts a wsl.exe
# client through WMI (outside the SSH job object, no admin rights needed) that
# waits on the target PID and exits by itself when that process ends.
#
# Usage (source it, then call):  start_wsl_keepalive "$$"
# Outside WSL, or without Windows interop, it is a no-op.

_select_wsl_interop() {
  # A tmux server keeps the WSL_INTEROP socket of the session that started it; once that
  # session closes, calling Windows binaries fails. Pick the first socket that still answers.
  local cmd=/mnt/c/Windows/System32/cmd.exe candidate
  for candidate in ${WSL_INTEROP:-} $(ls -t /run/WSL/*_interop 2>/dev/null); do
    [[ -S "$candidate" ]] || continue
    if WSL_INTEROP="$candidate" timeout 15 "$cmd" /c exit 0 >/dev/null 2>&1; then
      export WSL_INTEROP="$candidate"
      return 0
    fi
  done
  return 1
}

start_wsl_keepalive() {
  local target_pid="${1:?missing target PID}"
  local distro="${WSL_DISTRO_NAME:-}"
  local powershell=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
  if [[ -z "$distro" || ! -x "$powershell" ]]; then
    echo "wsl_keepalive: not in WSL or no Windows interop; skipped"
    return 0
  fi
  if ! _select_wsl_interop; then
    echo "wsl_keepalive: no live WSL interop socket; the run may die when all WSL sessions close" >&2
    return 1
  fi
  local command_line="wsl.exe -d $distro --exec tail --pid=$target_pid -f /dev/null"
  local result
  result=$("$powershell" -NoProfile -NonInteractive -Command \
    "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='$command_line'}; \"\$(\$r.ReturnValue) \$(\$r.ProcessId)\"" \
    2>/dev/null | tr -d '\r' | tail -n 1)
  if [[ "$result" =~ ^0\ ([0-9]+)$ ]]; then
    echo "wsl_keepalive: Windows PID ${BASH_REMATCH[1]} holds distro $distro until PID $target_pid exits"
    return 0
  fi
  echo "wsl_keepalive: failed to start keep-alive (got '$result'); the run may die when all WSL sessions close" >&2
  return 1
}
