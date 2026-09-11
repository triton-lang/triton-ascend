#!/bin/bash

SHELL_DIR="$(dirname "${BASH_SOURCE:-$0}")"
INSTALL_PATH="$(cd "${SHELL_DIR}" && pwd)"
TOTAL_RET="0"

uninstall_package() {
    local path="$1"
    local ret

    cd "${INSTALL_PATH}/${path}"
    ./uninstall.sh
    ret="$?" && [ ${ret} -ne 0 ] && TOTAL_RET="1"
    return ${ret}
}

if [ ! "$*" = "" ]; then
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$cur_date] [ERROR]: $*, parameter is not supported."
    exit 1
fi

exit ${TOTAL_RET}
