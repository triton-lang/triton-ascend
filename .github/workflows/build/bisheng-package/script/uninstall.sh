#!/bin/bash
# 此处定义各种变量
readonly PACKAGE_SHORT_NAME="ascendnpu-ir"
readonly PACKAGE_ARCH=$(arch)
readonly PACKAGE_ARCH_OS=$PACKAGE_ARCH-linux
INSTALL_DIRECTORY="tools"
INSTALL_SHARE_DIRECTORY="share/info/ascendnpu-ir"
upgrade_flag=n

tools_dir="${INSTALL_DIRECTORY}"

# 路径 xxxx/bisheng-toolkit
install_path="$(dirname $(
    cd "$(dirname "$0")/../../.."
    pwd
))"

config_file_path="${install_path}/${INSTALL_SHARE_DIRECTORY}/ascend-${PACKAGE_SHORT_NAME}_install.info"
version_file_path="${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info"
scene_file_path="${install_path}/${INSTALL_SHARE_DIRECTORY}/scene.info"

function print() {
    # 将关键信息打印到屏幕上
    echo "[AscendNPU-IR] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
}

#安全删除文件
function rm_file_safe() {
    local file_path=$1
    # 判断变量是否为空
    if [ -n "${file_path}" ]; then
        # 判断是否是文件
        if [ -f "${file_path}" ] || [ -h "${file_path}" ]; then
            rm -f "${file_path}"
            print "INFO" "delete file ${file_path} successfully"
        else
            print "WARNING" "the file ${file_path} is not exist"
        fi
    else
        print "WARNING" "the file ${file_path} path is NULL"
    fi
}

#安全删除文件夹
function rm_dir_safe() {
    local dir_path=$1
    # 判断变量不为空且不是系统根盘
    if [ -n "${dir_path}" ] && [[ ! "${dir_path}" =~ ^/+$ ]]; then
        # 判断是否是目录
        if [ -d "${dir_path}" ]; then
            rm -rf "${dir_path}"
            print "INFO" "delete directory ${dir_path} successfully"
        else
            print "WARNING" "the directory ${dir_path} is not exist"
        fi
    else
        print "WARNING" "the directory ${dir_path} path is NULL"
    fi
}

function delete_empty_folder() {
    if [ -d "${1}" ]; then
        if [ ! "$(ls -A ${1})" ]; then
            rm_dir_safe ${1}
        fi
    fi
}

# 更改目录下文件权限实施修改
chmod_to_modify() {
    chmod 755 -R $install_path 2> /dev/null
}

function __remove_uninstall_package() {
    local uninstall_file=$1
    if [ -f "${uninstall_file}" ]; then
        sed -i "\|uninstall_package \"${INSTALL_SHARE_DIRECTORY}/script\"|d" "${uninstall_file}"
        if [ $? -ne 0 ]; then
            print "ERROR" "remove ${uninstall_file} uninstall_package command failed!"
            exit 1
        fi
    fi
    num=$(grep "^uninstall_package " ${uninstall_file} | wc -l)
    if [ ${num} -eq 0 ]; then
        rm -f "${uninstall_file}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            print "ERROR" "delete file: ${uninstall_file}failed, please delete it by yourself."
        fi
    fi
}

function unregist_uninstall() {
    if [ -f "${totals_vresion_path}/cann_uninstall.sh" ]; then
        chmod u+w ${totals_vresion_path}/cann_uninstall.sh
        __remove_uninstall_package "${totals_vresion_path}/cann_uninstall.sh"
        if [ -f "${totals_vresion_path}/cann_uninstall.sh" ]; then
            chmod u-w ${totals_vresion_path}/cann_uninstall.sh
        fi
    fi
}

function deal_install_dir() {
    rm_dir_safe ${install_path}/${INSTALL_SHARE_DIRECTORY}
    rm_dir_safe ${install_path}/${tools_dir}/bishengir
    rm_file_safe ${install_path}/${PACKAGE_ARCH_OS}/bin/bishengir-compile
    rm_file_safe ${install_path}/${PACKAGE_ARCH_OS}/bin/hivmc-a5
    rm_file_safe ${install_path}/${PACKAGE_ARCH_OS}/bin/bishengir-opt
}

function deal_uninstall() {
    deal_install_dir
    totals_vresion_path=${install_path}
    unregist_uninstall
    delete_empty_folder "${totals_vresion_path}/${PACKAGE_ARCH_OS}/bin"
    delete_empty_folder "${totals_vresion_path}/${PACKAGE_ARCH_OS}"
    delete_empty_folder "${totals_vresion_path}/${INSTALL_SHARE_DIRECTORY}"
    delete_empty_folder "${totals_vresion_path}/share/info"
    delete_empty_folder "${totals_vresion_path}/share"
    delete_empty_folder "${totals_vresion_path}/tools"
}

# 程序开始
function main() {
    upgrade_flag=$1
    chmod_to_modify
    deal_uninstall
}

main $*
