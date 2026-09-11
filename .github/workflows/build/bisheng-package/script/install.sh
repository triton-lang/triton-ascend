#!/bin/bash
# Original script extracted from the ascendnpu-ir OBS run package
# (ascendnpu-ir_1.2.0_linux-x86-pr_2863.run, built with Makeself 2.5.0 +
# ASCEND_RUN_PACKAGE label). Two adaptations for packaging with stock makeself:
#   1. parse_script_args reads $1 instead of $3 (the original custom makeself
#      header prepended two synthetic args before the user's args; stock
#      makeself forwards the user's args verbatim).
#   2. BASE_PACKAGE_VERSION / PACKAGE_VERSION and the TARGET_BUILD_ARCH
#      placeholder are substituted at packaging time by bisheng-build.yml.
# 此处定义各种变量
readonly PACKAGE_SHORT_NAME="ascendnpu-ir"
readonly RUN_DIR_NAME="run_package"
readonly PACKAGE_ARCH=$(arch)
readonly PACKAGE_ARCH_OS=$PACKAGE_ARCH-linux
readonly TARGET_ARCH=TARGET_BUILD_ARCH
readonly CURRENT_ARCH=$(uname -m)
PACKAGE_NAME="AscendNPU-IR"
INSTALL_DIRECTORY="tools"
INSTALL_SHARE_DIRECTORY="share/info/ascendnpu-ir"
BASE_PACKAGE_VERSION="1.1.0"
PACKAGE_VERSION="1.1.0"
DEFAULT_INSTALL_PATH=""
username=$(id -nu)
usergroup=$(id -ng)

# 由输入命令行决定的参数
install_path=""
install_flag=n
is_create_install_path=n
uninstall_flag=n
input_path_flag=n
devel_flag=n
quiet_flag=n
install_for_all_flag=n
install_path_cmd="--install-path"
install_cmd=""
uninstall_path_cmd="--uninstall"
upgrade_path_cmd="--upgrade"


tools_dir="${INSTALL_DIRECTORY}"
is_create_arch_path=n
tools_path=""
arch_path=""

# 设置安装默认目录
if [ "$UID" = "0" ]; then
    # root用户安装时，默认选择install_for_all
    install_for_all_flag=y
    DEFAULT_INSTALL_PATH="/usr/local/Ascend"
else
    DEFAULT_INSTALL_PATH="${HOME}/Ascend"
fi

#日志文件的位置
if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
log_file="${log_dir}/ascend_install.log"

###  公用函数
function print_usage() {
    echo "Please use this option for more help: --help / -h"
    exit 1
}

function print_error(){
    print "ERROR" "Unsupported parameters : $1"
    print_usage
}

# 创建文件夹
create_folder() {
    if [ ! -d "$log_dir" ]; then
        mkdir -p $log_dir
    fi
}

# 将日志打印
function log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_=$1
    local msg_=$2
    local log_format_="[AscendNPU-IR] [$cur_date_] [$log_type_]: ${msg_}"
    if [ ! -f "$log_file" -a "$quiet_flag" = n ]; then
        echo $log_format_
    elif [ -f "$log_file" ]; then
        echo $log_format_ >>$log_file
    fi
}

function print() {
    if [ "$quiet_flag" = y -a "$1" = "INFO" ]; then
        log "$1" "$2"
        return
    fi
    # 将关键信息打印到屏幕上
    if [ ! -f "$log_file" ]; then
        echo "[AscendNPU-IR] [$(date +"%Y-%m-%d %H:%M:%S")] [$1] $2"
    else
        echo "[AscendNPU-IR] [$(date +"%Y-%m-%d %H:%M:%S")] [$1] $2" | tee -a $log_file
    fi
}

function file_check() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

function mkdir_dir() {
    #创建目录
    if [ ! -d "$1" ]; then
        mkdir -p $1
        if [ $? -ne 0 ]; then
            print "ERROR" "mkdir install path $1 permission denied"
            exit 1
        fi
        print "INFO" "mkdir install path $1 successfully"
    fi

    if [ "$install_for_all_flag" = n ]; then
        chmod 755 $1 2>/dev/null
    else
        chmod 755 $1 2>/dev/null
    fi
}

function get_ver() {
    #获取版本信息
    local ver
    ver=`grep Version $1 | cut -d '=' -f 2`
    if [ $? != 0 ]; then
        print "ERROR" "grep the version.info fail"
        exit 1
    fi
    echo $ver
}

function cp_file() {
    #拷贝文件
    cp -f $1 $2
    if [ $? = 0 ]; then
        print "INFO" "copy $3 to $2 successfully"
    else
        print "ERROR" "copy $3 to $2 fail"
        exit 1
    fi
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
            print "WARNING" "the file is not exist"
        fi
    else
        print "WARNING" "the file path is NULL"
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
            print "WARNING" "the directory is not exist"
        fi
    else
        print "WARNING" "the directory path is NULL"
    fi
}

# 递归授权
chmod_recur() {
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod $2 {} \; 2> /dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -exec chmod $2 {} \; 2> /dev/null
    fi
}

# 安装结束之后更改权限
chmod_after_install() {
    if [ "$UID" = "0" ] && [ "$install_for_all_flag" = y ]; then
        chmod 555 $tools_path/bishengir
        chmod -R 555 $tools_path/bishengir/*
    else
        chmod 550 $tools_path/bishengir
        chmod -R 550 $tools_path/bishengir/*
    fi
    chmod a-w $tools_path
}

# 提醒环境变量
prompt_set_env() {
    echo "Please make sure that
        - PATH includes $1/tools/bishengir/bin"
}

# 解析脚本自身的参数
function parse_script_args() {
    while true; do
        case "$1" in
        --check)
            exit 0
            ;;
        --help | -h)
            print_usage
            ;;
        --version)
            echo "${PACKAGE_SHORT_NAME} ${PACKAGE_VERSION}"
            exit 0
            ;;
        --install)
            install_flag=y
            shift
            ;;
        --install-path=*)
            # 去除指定安装目录后所有的 "/"
            local temp_path=$(echo $1 | cut -d"=" -f2 | sed "s/\/*$//g")
            # path只支持绝对路径
            if [[ "${temp_path}" =~ ^/.* ]]; then
                install_path=${temp_path}
            else
                print "ERROR" "parameter error $3, must absolute path"
                exit 1
            fi
            input_path_flag=y
            shift
            ;;
        --uninstall)
            uninstall_flag=y
            shift
            ;;
        --devel)
            devel_flag=y
            shift
            ;;
        --upgrade)
            upgrade_flag=y
            shift
            ;;
        --quiet)
            quiet_flag=y
            shift
            ;;
        --run)
            install_flag=y
            shift
            ;;
        --full)
            install_flag=y
            shift
            ;;
        --install-for-all)
            install_for_all_flag=y
            shift
            ;;
        -*)
            print_error "$3"
            ;;
        *)
            break
            ;;
        esac
    done
}

### 脚本入参的相关处理函数
function check_script_args() {
    ######################  check params confilct ###################
    if [ $# -lt 3 ]; then
        print_usage
    fi
    local args_num=0
    if [ "$uninstall_flag" = y ]; then
        let 'args_num+=1'
    fi
    if [ "$upgrade_flag" = y ]; then
        let 'args_num+=1'
    fi
    if [ "$devel_flag" = y ]; then
        let 'args_num+=1'
    fi
    if [ "$install_flag" = y ]; then
        let 'args_num+=1'
    fi
    # 检测脚本参数的组合关系
    if [ $args_num -lt 1 ] || [ $args_num -gt 1 ]; then
        print "ERROR" "Unsupported parameters, operation failed."
        exit 1
    fi
    if [ "$input_path_flag" = y ]; then
        if [ "${uninstall_flag}" = "n" ] && [ "$install_flag" = "n" ] && [ "$upgrade_flag" = "n" ] && [ "${devel_flag}" = "n" ]; then
            print "ERROR" "Unsupported separate 'install-path' used independently"
            exit 1
        fi
    fi
}

function complete_params() {
    # 补齐具体执行安装，升级，卸载等流程需要的参数，比如升级时版本号的确认
    local tmp_install_path=${DEFAULT_INSTALL_PATH}
    if [ "$input_path_flag" = "y" ]; then
        tmp_install_path=${install_path}
    fi

    install_path="${tmp_install_path}"
    tools_path="${install_path}/${INSTALL_DIRECTORY}"
    arch_path="${install_path}/${PACKAGE_ARCH_OS}"
    config_file_path="${install_path}/${INSTALL_SHARE_DIRECTORY}/ascend-${PACKAGE_SHORT_NAME}_install.info"
    version_file_path="${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info"
    scene_file_path="${install_path}/${INSTALL_SHARE_DIRECTORY}/scene.info"
}

function log_init() {
    # 日志模块初始化
    # 判断输入的安装路径路径是否存在，不存在则创建
    if [ ! -f "$log_file" ]; then
        touch $log_file
        if [ $? -ne 0 ]; then
            print "ERROR" "touch $log_file permission denied"
            exit 1
        fi
    fi
    chmod 640 $log_file
    if [ "${install_flag}" = y ]; then
        print "INFO" "install start"
    elif [ "${uninstall_flag}" = y ]; then
        print "INFO" "uninstall start"
    elif [ "${upgrade_flag}" = y ]; then
        print "INFO" "upgrade start"
    elif [ "${rollback_flag}" = y ]; then
        print "INFO" "rollback start"
    fi
}

# 执行安装run包
function deal_install() {
    check_binary_architecture
    if [ "${install_flag}" = y ] || [ "${devel_flag}" = y ] || [ "${upgrade_flag}" = y ]; then
        if [ ! -d "$install_path" ]; then
            mkdir_dir "$install_path"
            is_create_install_path=y
            if [ "$UID" = "0" ] && [ "$install_for_all_flag" = y ]; then
                chmod 755 $install_path
            else
                chmod 750 $install_path
            fi
        fi
    fi

    if [ ! -d "$tools_path" ]; then
        mkdir_dir "$tools_path"
        if [ "$UID" = "0" ] && [ "$install_for_all_flag" = y ]; then
            chmod 755 $tools_path
        else
            chmod 750 $tools_path
        fi
    fi

    if [ ! -d "${install_path}/${INSTALL_SHARE_DIRECTORY}" ]; then
        mkdir_dir "${install_path}/${INSTALL_SHARE_DIRECTORY}"
        mkdir_dir "${install_path}/${INSTALL_SHARE_DIRECTORY}/bin"
        if [ "$UID" = "0" ] && [ "$install_for_all_flag" = y ]; then
            chmod 755 ${install_path}/${INSTALL_SHARE_DIRECTORY}
            chmod 755 ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin
        else
            chmod 750 ${install_path}/${INSTALL_SHARE_DIRECTORY}
            chmod 750 ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin
        fi
    fi

    if [ ! -d "${install_path}/${PACKAGE_ARCH_OS}" ]; then
        mkdir_dir "${install_path}/${PACKAGE_ARCH_OS}"
        is_create_arch_path=y
        if [ ! -d "${install_path}/${PACKAGE_ARCH_OS}/bin" ]; then
          mkdir_dir "${install_path}/${PACKAGE_ARCH_OS}/bin"
        fi
        if [ "$UID" = "0" ] && [ "$install_for_all_flag" = y ]; then
            chmod 755 ${install_path}/${PACKAGE_ARCH_OS}
        else
            chmod 750 ${install_path}/${PACKAGE_ARCH_OS}
        fi
    fi

    chmod -R +w ${tools_path}

    cp -af ./bishengir ${tools_path}/
    cp -af ./script ${install_path}/${INSTALL_SHARE_DIRECTORY}/
    cp -af ./set_env.sh ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin

    is_arch_writable=n
    is_arch_bin_writable=n
    if [ -w "${install_path}/${PACKAGE_ARCH_OS}" ]; then
        is_arch_writable=y
    fi
    if [ -w "${install_path}/${PACKAGE_ARCH_OS}/bin" ]; then
        is_arch_bin_writable=y
    fi
    chmod u+w ${install_path}/${PACKAGE_ARCH_OS}
    chmod u+w ${install_path}/${PACKAGE_ARCH_OS}/bin
    if [ -L "${install_path}/${PACKAGE_ARCH_OS}/bishengir-compile" ]; then
        rm_file_safe ${install_path}/${PACKAGE_ARCH_OS}/bishengir-compile
    fi
    if [ -L "${install_path}/${PACKAGE_ARCH_OS}/hivmc-a5" ]; then
        rm_file_safe ${install_path}/${PACKAGE_ARCH_OS}/hivmc-a5
    fi
    if [ -L "${install_path}/${PACKAGE_ARCH_OS}/bishengir-opt" ]; then
        rm_file_safe ${install_path}/${PACKAGE_ARCH_OS}/bishengir-opt
    fi
    ln -sf ../../${tools_dir}/bishengir/bin/bishengir-compile ${install_path}/${PACKAGE_ARCH_OS}/bin/bishengir-compile
    ln -sf ../../${tools_dir}/bishengir/bin/hivmc-a5 ${install_path}/${PACKAGE_ARCH_OS}/bin/hivmc-a5
    ln -sf ../../${tools_dir}/bishengir/bin/bishengir-opt ${install_path}/${PACKAGE_ARCH_OS}/bin/bishengir-opt

    CURRENT_VERSION=$BASE_PACKAGE_VERSION

    if [ "$is_arch_writable" = n ]; then
        chmod u-w ${install_path}/${PACKAGE_ARCH_OS}
    fi
    if [ "$is_arch_bin_writable" = n ]; then
        chmod u-w ${install_path}/${PACKAGE_ARCH_OS}/bin
    fi
}

# 执行删除run包
function deal_uninstall() {
    ${install_path}/${INSTALL_SHARE_DIRECTORY}/script/uninstall.sh
}

#移除卸载脚本uninstall_package
function remove_cann_uninstall() {
    if [ -f "${install_path}/cann_uninstall.sh" ]; then
        sed -i "/uninstall_package \"share\/info\/ascendnpu-ir\/script\"/d" "${install_path}/cann_uninstall.sh"
        if [ $? -ne 0 ]; then
            print "ERROR" "remove ${install_path}/cann_uninstall.sh uninstall_package command failed"
            exit 2
        fi
    fi
}

function write_cann_uninstall() {
    chmod 500 "${install_path}/cann_uninstall.sh"
    chmod u+w "${install_path}/cann_uninstall.sh"
    (grep "${INSTALL_SHARE_DIRECTORY}/script" "${install_path}/cann_uninstall.sh") &> /dev/null
    if [ $? -eq 0 ]; then
        remove_cann_uninstall
    fi
    sed -i "/^exit /i uninstall_package \"${INSTALL_SHARE_DIRECTORY}/script\"" "${install_path}/cann_uninstall.sh"
    chmod 500 "${install_path}/cann_uninstall.sh"
}

function regist_uninstall() {
    if [ -f "${install_path}/cann_uninstall.sh" ]; then
        write_cann_uninstall
    else
        cp -af script/cann_uninstall.sh ${install_path}
        write_cann_uninstall
    fi
}

function upgrade_uninstall() {
    if [ -e "${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info" ]; then
        ${install_path}/${INSTALL_SHARE_DIRECTORY}/script/uninstall.sh ${upgrade_flag}
    fi
}

function deal_with_packages() {
    # 安装、卸载、升级
    if [ "$1" == "install" ]; then
        deal_install
        upgrade_config_file "install"
        prompt_set_env ${install_path}
        deal_with_env ${install_path}
        regist_uninstall
        chmod_after_install
    elif [ "$1" == "uninstall" ]; then
        deal_uninstall
    elif [ "$1" == "upgrade" ]; then
        upgrade_uninstall
        deal_install
        upgrade_config_file "upgrade"
        regist_uninstall
        chmod_after_install
    fi
    print "INFO" "${PACKAGE_NAME}-${BASE_PACKAGE_VERSION} ${1} success"
    exit 0
}

function deal_with_env() {
  touch ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin/set_env.sh
  chmod +w ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin/set_env.sh
  local safe_path="$(printf '%s' "$install_path" | sed 's/[\/&]/\\&/g')"
  cat > ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin/set_env.sh <<'TEMPLATE'
#!/bin/sh
export PATH="INSTALL_PATH/tools/bishengir/bin:${PATH}"
TEMPLATE
  sed -i "s|INSTALL_PATH|$safe_path|g" ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin/set_env.sh
  chmod +x ${install_path}/${INSTALL_SHARE_DIRECTORY}/bin/set_env.sh
}

function upgrade_config_file() {
    # 安装、升级、卸载
    if [ "$1" == "install" ] || [ "$1" == "upgrade" ]; then
        echo "Version=${BASE_PACKAGE_VERSION}" >${version_file_path}
        log "INFO" "version=${BASE_PACKAGE_VERSION}"

        echo "path=${install_path}" >${config_file_path}
        log "INFO" "path=${install_path}"
        echo "arch=${PACKAGE_ARCH}" >>${config_file_path}
        log "INFO" "arch=${PACKAGE_ARCH}"
        echo "ascendnpu-ir_username=${username}" >>${config_file_path}
        echo "ascendnpu-ir_usergroup=${usergroup}" >>${config_file_path}

        echo "os=linux" >${scene_file_path}
        echo "arch=${PACKAGE_ARCH}" >>${scene_file_path}
        log "INFO" "arch=${PACKAGE_ARCH}"
    fi
}

### 安装，卸载，升级 流程
function install_process() {
    # 安装
    deal_with_packages "install"
}

function reinstall_check() {
    if [ -e "${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info" ]; then
        return 0
    else
        return 1
    fi
}

function upgrade_process() {
    # 升级过程除了路径不需要输入理论上与安装一样
    # 各种检测
    reinstall_check
    if [ $? -eq 1 ]; then
        print "ERROR" "run package is not installed, upgrade failed"
        print "ERROR" "check the environment failed"
        exit 2
    fi

    # 安装
    deal_with_packages "upgrade"
}

function uninstall_check() {
    if [ -d "${tools_path}" ]; then
        return 0
    else
        return 1
    fi
}

function uninstall_process() {
    # 各种检测
    uninstall_check
    if [ $? -eq 1 ]; then
        print "ERROR" "run package is not installed, uninstall failed"
        print "ERROR" "check the environment failed"
        exit 2
    fi
    # 卸载
    deal_with_packages "uninstall"
}

function get_current_version() {
    if [ -d "${install_path}/${INSTALL_SHARE_DIRECTORY}" ]; then
        if [ -f "${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info" ]; then
            cur_ver=`get_ver "${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info"`
        else
            print "WARNING" "the file ${install_path}/${INSTALL_SHARE_DIRECTORY}/version.info is not exist"
            exit 0
        fi
    else
        print "WARNING" "the dir ${install_path}/${INSTALL_SHARE_DIRECTORY}/ is not exist"
        exit 0
    fi
}

function process() {
    if [ "$install_flag" = "y" ] || [ "${devel_flag}" = "y" ]; then
        install_process
    elif [ "$upgrade_flag" = "y" ]; then
        upgrade_process
    elif [ "$uninstall_flag" = "y" ]; then
        uninstall_process
    fi
}

function create_install_dir() {
    local install_path="$1"
    if [ -d "${install_path}" ]; then
        return
    fi
    local tmp_install_path="$1"
    while [ ! -d $(dirname ${tmp_install_path}) ]; do
        tmp_install_path=$(dirname ${tmp_install_path})
    done
    mkdir -p ${install_path}
    if [ $? -ne 0 ]; then
        print "ERROR" "mkdir insatll path ${install_path} permission denied"
        exit 1
    fi
    if [ "$UID" != "0" ] && [ "$install_for_all_flag" = n ]; then
        chmod -R 755 ${tmp_install_path}
    else
        chmod -R 755 ${tmp_install_path}
    fi
}

function check_binary_architecture() {
    case "$TARGET_ARCH" in
      "x86")
          if [ "$CURRENT_ARCH" != "x86_64" ]; then
              echo "WARNING" "the architecture of the package x86_64 may not be compatible with the system $CURRENT_ARCH !"
          fi
          ;;
      "aarch64")
          if [ "$CURRENT_ARCH" != "aarch64" ]; then
              echo "WARNING" "the architecture of the package aarch64 may not be compatible with the system $CURRENT_ARCH !"
          fi
          ;;
    esac
}

# 程序开始
function main() {
    umask 0022
    create_folder
    parse_script_args $*
    # check_script_args $*
    complete_params
    log_init
    create_install_dir "${install_path}"
    process
}

main $*
